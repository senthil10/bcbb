#!/usr/bin/env python
"""Script to check for finalized illumina runs and report to messaging server.

Run this script with an hourly cron job; it looks for newly finished output
directories for processing.

Usage:
    illumina_finished_msg.py <YAML local config>
                             [<post-processing config file>]

Supplying a post-processing configuration file skips the messaging step and
we moves directly into analysis processing on the current machine. Use
this if there is no RabbitMQ messaging server and your dump machine is directly
connected to the analysis machine. You will also want to set postprocess_dir in
the YAML local config to the directory to write fastq and analysis files.

The Galaxy config needs to have information on the messaging server and queues.

The local config should have the following information:

    dump_directories: directories to check for machine output
    msg_db: flat file of reported output directories
"""
import os
import operator
import socket
import glob
import getpass
import subprocess
import time
import sys
from optparse import OptionParser
import xml.etree.ElementTree as ET
import re
import csv
from shutil import copyfile
from multiprocessing import Pool
from itertools import izip

import logbook

from bcbio.solexa import samplesheet
from bcbio.log import create_log_handler, logger2
from bcbio import utils
from bcbio.distributed import messaging
from bcbio.solexa.flowcell import (get_flowcell_info, get_fastq_dir, get_qseq_dir)
from bcbio.pipeline.config_loader import load_config

# Import functionality to convert MiSeq samplesheets from the scilifelab repo (if available)
try:
    from scilifelab.illumina.miseq import MiSeqRun
except ImportError:
    pass

LOG_NAME = os.path.splitext(os.path.basename(__file__))[0]
log = logbook.Logger(LOG_NAME)


def main(*args, **kwargs):
    local_config = args[0]
    post_process_config = args[1] if len(args) > 1 else None
    kwargs["post_process_config"] = post_process_config
    config = load_config(local_config)

    log_handler = create_log_handler(config, True)
    with log_handler.applicationbound():
        search_for_new(config, local_config, **kwargs)


def search_for_new(*args, **kwargs):
    """Search for any new unreported directories.
    """
    config = args[0]
    reported = _read_reported(config["msg_db"])
    for dname in _get_directories(config):
        # Only process a directory if it isn't listed in the transfer db or if it was specifically requested
        # on the command line
        if os.path.isdir(dname) and \
        ((kwargs.get("run_id",None) is None and not any(dir.startswith(dname) for dir in reported)) or \
         kwargs.get("run_id",None) == os.path.basename(dname)):

            # Injects run_name on logging calls.
            # Convenient for run_name on "Subject" for email notifications
            def run_setter(record):
                return record.extra.__setitem__('run', os.path.basename(dname))

            with logbook.Processor(run_setter):
                if kwargs.get("post_process_only",False):
                    loc_args = (dname, ) + args + (None, )
                    _post_process_run(*loc_args, **kwargs)
                    continue
                if _do_initial_processing(dname):
                    initial_processing(dname, *args, **kwargs)

                elif _do_first_read_processing(dname):
                    process_first_read(dname, *args, **kwargs)

                elif _do_second_read_processing(dname):
                    process_second_read(dname, *args, **kwargs)
                else:
                    pass

                # Re-read the reported database to make sure it hasn't
                # changed while processing.
                reported = _read_reported(config["msg_db"])
    
def initial_processing(*args, **kwargs):
    """Initial processing to be performed after the first base report
    """
    dname, config = args[0:2]
    # Touch the indicator flag that processing of read1 has been started
    utils.touch_indicator_file(os.path.join(dname, "initial_processing_started.txt"))

    # Copy the samplesheet to the run folder
    ss_file = samplesheet.run_has_samplesheet(dname, config)
    if ss_file:
        dst = os.path.join(dname,os.path.basename(ss_file))
        try:
            copyfile(ss_file,dst)
        except IOError, e:
            logger2.error("Error copying samplesheet {} from {} to {}: {}" \
                          "".format(os.path.basename(ss_file),
                                    os.path.dirname(ss_file),
                                    os.path.dirname(dst),
                                    e))
    # If this is a MiSeq run and we have the scilifelab modules loaded,
    # convert the MiSeq samplesheet into a format compatible with casava
    elif _is_miseq_run(dname):
        if 'scilifelab.illumina.miseq' in sys.modules:
            mrun = MiSeqRun(dname)
            hiseq_ssheet = os.path.join(dname,'{}.csv'.format(_get_flowcell_id(dname)))
            mrun.write_hiseq_samplesheet(hiseq_ssheet)
        # If the module wasn't loaded, there's nothing we can do, so warn
        else:
            logger2.error("The necessary dependencies for processing MiSeq runs with CASAVA could not be loaded")
    
    # Upload the necessary files
    loc_args = args + (None, )
    _post_process_run(*loc_args, **{"fetch_msg": kwargs.get("fetch_msg", False),
                                    "process_msg": False,
                                    "store_msg": kwargs.get("store_msg", False),
                                    "backup_msg": kwargs.get("backup_msg", False),
                                    "push_data": kwargs.get("push_data", False)})

    # Touch the indicator flag that processing of read1 has been completed
    utils.touch_indicator_file(os.path.join(dname, "initial_processing_completed.txt"))


def process_first_read(*args, **kwargs):
    """Processing to be performed after the first read and the index reads
    have been sequenced
    """
    dname, config = args[0:2]
    # Do bcl -> fastq conversion and demultiplexing using Casava1.8+
    if kwargs.get("casava", False):
        if not kwargs.get("no_casava_processing", False):
            logger2.info("Generating fastq.gz files for read 1 of {:s}".format(dname))

            # Touch the indicator flag that processing of read1 has been started
            utils.touch_indicator_file(os.path.join(dname, "first_read_processing_started.txt"))
            unaligned_dirs = _generate_fastq_with_casava(dname, config, r1=True)
            logger2.info("Done generating fastq.gz files for read 1 of {:s}".format(dname))

            # Extract the top barcodes from the undemultiplexed fraction
            for unaligned_dir in unaligned_dirs:
                if config["program"].get("extract_barcodes", None):
                    extract_top_undetermined_indexes(dname, unaligned_dir, config)

        for unaligned_dir in unaligned_dirs:
            unaligned_dir = os.path.join(dname, "Unaligned")
            loc_args = args + (unaligned_dir,)
            _post_process_run(*loc_args, **{"fetch_msg": kwargs.get("fetch_msg", False),
                                            "process_msg": False,
                                            "store_msg": kwargs.get("store_msg", False),
                                            "backup_msg": kwargs.get("backup_msg", False),
                                            "push_data": kwargs.get("push_data", False)})

        # Touch the indicator flag that processing of read1 has been completed
        utils.touch_indicator_file(os.path.join(dname, "first_read_processing_completed.txt"))


def process_second_read(*args, **kwargs):
    """Processing to be performed after all reads have been sequences
    """
    dname, config = args[0:2]
    logger2.info("The instrument has finished dumping on directory %s" % dname)

    utils.touch_indicator_file(os.path.join(dname, "second_read_processing_started.txt"))
    _update_reported(config["msg_db"], dname)
    fastq_dir = None

    # Do bcl -> fastq conversion and demultiplexing using Casava1.8+
    if kwargs.get("casava", False):
        if not kwargs.get("no_casava_processing", False):
            logger2.info("Generating fastq.gz files for {:s}".format(dname))
            _generate_fastq_with_casava(dname, config)

    else:
        _process_samplesheets(dname, config)
        if kwargs.get("qseq", True):
            logger2.info("Generating qseq files for {:s}".format(dname))
            _generate_qseq(get_qseq_dir(dname), config)

        if kwargs.get("fastq", True):
            logger2.info("Generating fastq files for {:s}".format(dname))
            fastq_dir = _generate_fastq(dname, config)
            if kwargs.get("remove_qseq", False):
                _clean_qseq(get_qseq_dir(dname), fastq_dir)

            _calculate_md5(fastq_dir)

    # Call the post_processing method
    loc_args = args + (fastq_dir,)
    _post_process_run(*loc_args, **{"fetch_msg": kwargs.get("fetch_msg", False),
                                    "process_msg": kwargs.get("process_msg", False),
                                    "store_msg": kwargs.get("store_msg", False),
                                    "backup_msg": kwargs.get("backup_msg", False),
                                    "push_data": kwargs.get("push_data", False)})

    # Update the reported database after successful processing
    _update_reported(config["msg_db"], dname)
    utils.touch_indicator_file(os.path.join(dname, "second_read_processing_completed.txt"))


def extract_top_undetermined_indexes(fc_dir, unaligned_dir, config):
    """Extract the top N=25 barcodes from the undetermined indices output
    """
    infile_glob = os.path.join(unaligned_dir, "Undetermined_indices", "Sample_lane*", "*_R1_*.fastq.gz")
    infiles = glob.glob(infile_glob)
    
    # Only run as many simultaneous processes as number of cores specified in config
    procs = []
    num_cores = config["algorithm"].get("num_cores", 1)

    # Iterate over the infiles and process each one
    while len(infiles) > 0:
        # Wait one minute if we are already using the maximum amount of cores
        if len([p for p in procs if p[0].poll() is None]) == num_cores:
            time.sleep(60)

        else:
            infile = infiles.pop()
            fname = os.path.basename(infile)

            # Parse the lane number from the filename
            m = re.search(r'_L0*(\d+)_', fname)
            if len(m.groups()) == 0:
                raise ValueError("Could not determine lane from filename {:s}".format(fname))

            lane = m.group(1)

            # Open a subprocess for the extraction, writing output and errors to a metric file
            logger2.info("Extracting top indexes from lane {:s}".format(lane))
            metricfile = os.path.join(fc_dir, fname.replace("fastq.gz",
                                                            "undetermined_indices_metrics"))
            fh = open(metricfile, "w")
            cl = [config["program"]["extract_barcodes"], infile, lane,
                  '--nindex', 10]
            p = subprocess.Popen([str(c) for c in cl], stdout=fh, stderr=fh)
            procs.append([p, fh, metricfile])

    # Wait until all running processes have finished
    while len([p for p in procs if p[0].poll() is None]) > 0:
        time.sleep(60)

    # Parse all metricfiles into one list of dicts
    logger2.info("Merging lane metrics into one flowcell metric")
    metrics = []
    header = []
    for p in procs:
        # Close the filehandle
        p[1].close()

        # Parse the output into a dict using a DictReader
        with open(p[2]) as fh:
            c = csv.DictReader(fh, dialect=csv.excel_tab)
            header = c.fieldnames
            for row in c:
                metrics.append(row)
        # Remove the metricfile
        os.unlink(p[2])

    # Write the metrics to one output file
    fcid = _get_flowcell_id(fc_dir)
    metricfile = os.path.join(unaligned_dir, "Basecall_Stats_{}".format(fcid), "Undemultiplexed_stats.metrics")
    with open(metricfile, "w") as fh:
        w = csv.DictWriter(fh, fieldnames=header, dialect=csv.excel_tab)
        w.writeheader()
        w.writerows(metrics)

    logger2.info("Undemultiplexed metrics written to {:s}".format(metricfile))
    return metricfile


def _post_process_run(dname, config, config_file, fastq_dir, **kwargs):
    """With a finished directory, send out message or process directly.
    """
    post_config_file = kwargs.get("post_config_file", None)

    # without a configuration file, send out message for processing
    if post_config_file is None:
        push_data = kwargs.get("push_data", False)
        fetch_msg = kwargs.get("fetch_msg", False)
        process_msg = kwargs.get("process_msg", False)
        store_msg = kwargs.get("store_msg", False)
        backup_msg = kwargs.get("backup_msg", False)

        run_module = "bcbio.distributed.tasks"
        store_files, process_files, backup_files = _files_to_copy(dname)

        if push_data:
            data = {"directory": dname, "to_copy": process_files}
            simple_upload(config, data)

        if push_data and process_msg:
            config["pushed"] = True
            finished_message("analyze", run_module, dname,
                             process_files, config, config_file, pushed=True)

        if process_msg and not push_data:
            finished_message("analyze_and_upload", run_module, dname,
                             process_files, config, config_file)
        elif fetch_msg:
            finished_message("fetch_data", run_module, dname,
                             process_files, config, config_file)
        if store_msg:
            raise NotImplementedError("Storage server needs update.")
            finished_message("long_term_storage", run_module, dname,
                             store_files, config, config_file)
        if backup_msg:
            finished_message("backup_data", run_module, dname,
                             backup_files, config, config_file)

    # otherwise process locally
    else:
        analyze_locally(dname, post_config_file, fastq_dir)


def simple_upload(remote_info, data):
    """Upload generated files to specified host using rsync
    """
    include = ['--include=*/']
    for fcopy in data['to_copy']:
        include.extend(["--include", "{}**/*".format(fcopy)])
        include.append("--include={}".format(fcopy))
    # By including both these patterns we get the entire directory
    # if a directory is given, or a single file if a single file is
    # given.

    cl = ["rsync", \
          "--checksum", \
          "--archive", \
          "--partial", \
          "--progress", \
          "--prune-empty-dirs"
          ]

    # file / dir inclusion specification
    cl.extend(include)
    cl.append("--exclude=*")

    # source and target
    cl.extend([
          # source
          data["directory"], \
          # target
          "{store_user}@{store_host}:{store_dir}".format(**remote_info)
         ])

    logdir = remote_info.get("log_dir",os.getcwd())
    rsync_out = os.path.join(logdir,"rsync_transfer.out")
    rsync_err = os.path.join(logdir,"rsync_transfer.err")
    ro = open(rsync_out, 'a')
    re = open(rsync_err, 'a')
    try:
        ro.write("-----------\n{}\n".format(" ".join(cl)))
        re.write("-----------\n{}\n".format(" ".join(cl)))
        ro.flush()
        re.flush()
        subprocess.check_call(cl, stdout=ro, stderr=re)
    except subprocess.CalledProcessError, e:
        logger2.error("rsync transfer of {} FAILED with (exit code {}). " \
                      "Please check log files {:s} and {:s}".format(data["directory"],
                                                                    str(e.returncode),
                                                                    rsync_out,
                                                                    rsync_err))
        raise e
    finally:
        ro.close()
        re.close()

def analyze_locally(dname, post_config_file, fastq_dir):
    """Run analysis directly on the local machine.
    """
    assert fastq_dir is not None
    post_config = load_config(post_config_file)
    analysis_dir = os.path.join(fastq_dir, os.pardir, "analysis")
    utils.safe_makedir(analysis_dir)
    with utils.chdir(analysis_dir):
        if post_config["algorithm"]["num_cores"] == "messaging":
            prog = post_config["analysis"]["distributed_process_program"]
        else:
            prog = post_config["analysis"]["process_program"]
        cl = [prog, post_config_file, dname]
        run_yaml = os.path.join(dname, "run_info.yaml")
        if os.path.exists(run_yaml):
            cl.append(run_yaml)
        subprocess.check_call(cl)


def _process_samplesheets(dname, config):
    """Process Illumina samplesheets into YAML files for post-processing.
    """
    ss_file = samplesheet.run_has_samplesheet(dname, config)
    if ss_file:
        out_file = os.path.join(dname, "run_info.yaml")
        logger2.info("CSV Samplesheet %s found, converting to %s" %
                 (ss_file, out_file))
        samplesheet.csv2yaml(ss_file, out_file)


def _generate_fastq_with_casava_task(args):
    """Perform demultiplexing and generate fastq.gz files for the current
    flowecell using CASAVA (>1.8).
    """
    bp = args.get('bp')
    samples_group = args.get('samples')
    base_mask = samples_group['base_mask']
    samples = samples_group['samples']
    fc_dir = args.get('fc_dir')
    config = args.get('config')
    r1 = args.get('r1', False)
    ss = 'SampleSheet_{bp}bp.csv'.format(bp=str(bp))
    unaligned_folder = 'Unaligned_{bp}bp'.format(bp=str(bp))
    out_file = 'configureBclToFastq_{bp}bp.out'.format(bp=str(bp))
    err_file = 'configureBclToFastq_{bp}bp.err'.format(bp=str(bp))

    #Prepare CL arguments and call configureBclToFastq
    basecall_dir = os.path.join(fc_dir, "Data", "Intensities", "BaseCalls")
    casava_dir = config["program"].get("casava")
    unaligned_dir = os.path.join(fc_dir, unaligned_folder)
    samplesheet_file = os.path.join(fc_dir, ss)
    num_mismatches = config["algorithm"].get("mismatches", 1)
    num_cores = config["algorithm"].get("num_cores", 1)
    im_stats = config["algorithm"].get("ignore-missing-stats", False)
    im_bcl = config["algorithm"].get("ignore-missing-bcl", False)
    im_control = config["algorithm"].get("ignore-missing-control", False)

    # Write to log files
    configure_out = os.path.join(fc_dir, out_file)
    configure_err = os.path.join(fc_dir, err_file)
    casava_out = os.path.join(fc_dir, "bclToFastq_R{:d}.out".format(2 - int(r1)))
    casava_err = os.path.join(fc_dir, "bclToFastq_R{:d}.err".format(2 - int(r1)))

    cl = [os.path.join(casava_dir, "configureBclToFastq.pl")]
    cl.extend(["--input-dir", basecall_dir])
    cl.extend(["--output-dir", unaligned_dir])
    cl.extend(["--mismatches", str(num_mismatches)])
    cl.extend(["--fastq-cluster-count", "0"])
    if samplesheet_file is not None:
        cl.extend(["--sample-sheet", samplesheet_file])

    if im_stats:
        cl.append("--ignore-missing-stats")

    if im_bcl:
        cl.append("--ignore-missing-bcl")

    if im_control:
        cl.append("--ignore-missing-control")

    if base_mask is not None:
        cl.extend(["--use-bases-mask", ','.join(base_mask)])

    if r1:
        #Create separate samplesheet and folder
        with open(os.path.join(fc_dir, ss), 'w') as fh:
            samplesheet = csv.DictWriter(fh, fieldnames=samples['fieldnames'], dialect='excel')
            samplesheet.writeheader()
            samplesheet.writerows(samples['samples'])

        # Run configuration script
        logger2.info("Configuring BCL to Fastq conversion")
        logger2.debug(cl)

        co = open(configure_out, 'w')
        ce = open(configure_err, 'w')
        try:
            co.write("{}\n".format(" ".join(cl)))
            ce.write("{}\n".format(" ".join(cl)))
            subprocess.check_call(cl, stdout=co, stderr=ce)
        except subprocess.CalledProcessError, e:
            logger2.error("Configuring BCL to Fastq conversion for {:s} FAILED " \
                          "(exit code {}), please check log files {:s}, {:s}".format(fc_dir,
                                                                                     str(e.returncode),
                                                                                     configure_out,
                                                                                     configure_err))
            raise e
        finally:
            co.close()
            ce.close()

   # Go to <Unaligned> folder
    with utils.chdir(unaligned_dir):
        # Perform make
        cl = ["make", "-j", str(num_cores)]
        if r1:
            cl.append("r1")

        logger2.info("Demultiplexing and converting bcl to fastq.gz")
        logger2.debug(cl)

        co = open(casava_out, 'w')
        ce = open(casava_err, 'w')
        try:
            co.write("{}\n".format(" ".join(cl)))
            ce.write("{}\n".format(" ".join(cl)))
            subprocess.check_call(cl, stdout=co, stderr=ce)
        except subprocess.CalledProcessError, e:
            logger2.error("BCL to Fastq conversion for {:s} FAILED " \
                          "(exit code {}), please check log files {:s}, "\
                          "{:s}".format(fc_dir,
                                        str(e.returncode),
                                        casava_out,
                                        casava_err))
            raise e
        finally:
            co.close()
            ce.close()

    logger2.debug("Done")
    return unaligned_dir


def _generate_fastq_with_casava(fc_dir, config, r1=False):
    """Prepare and call the task to perform demultiplexing and generation of
    fastq.gz files for the current flowcell in using CASAVA (>1.8). If the
    number of cores specified is > 1, the demultiplexing will be done in
    parallel.
    """
        
    base_masks = _get_bases_mask(fc_dir)
    num_cores = config["algorithm"].get("num_cores", 1)
    #Prepare the list of arguments to call configureBclToFastq
    args_list = []
    [args_list.append({'bp': k, 'samples': v, 'fc_dir':fc_dir, 'config':config, 'r1':r1}) \
                        for k, v in base_masks.iteritems()]

    p = Pool(processes=num_cores)
    unaligned_dirs = p.map(_generate_fastq_with_casava_task, args_list)

    return unaligned_dirs


def _generate_fastq(fc_dir, config, compress_fastq):
    """Generate fastq files for the current flowcell.
    """
    fc_name, fc_date = get_flowcell_info(fc_dir)
    short_fc_name = "%s_%s" % (fc_date, fc_name)
    fastq_dir = get_fastq_dir(fc_dir)
    basecall_dir = os.path.split(fastq_dir)[0]
    postprocess_dir = config.get("postprocess_dir", "")
    if postprocess_dir:
        fastq_dir = os.path.join(postprocess_dir, os.path.basename(fc_dir), "fastq")

    if not fastq_dir == fc_dir:  # and not os.path.exists(fastq_dir):

        with utils.chdir(basecall_dir):
            lanes = sorted(list(set([f.split("_")[1] for f in
                glob.glob("*qseq.txt")])))
            cl = ["solexa_qseq_to_fastq.py", short_fc_name,
                  ",".join(lanes)]
            if postprocess_dir:
                cl += ["-o", fastq_dir]
            if compress_fastq:
                cl += ["--gzip"]

            logger2.debug("Converting qseq to fastq on all lanes.")
            subprocess.check_call(cl)

    return fastq_dir


def _calculate_md5(fastq_dir):
    """Calculate the md5sum for the fastq files
    """
    glob_str = "*_fastq.txt"
    fastq_files = glob.glob(os.path.join(fastq_dir, glob_str))

    md5sum_file = os.path.join(fastq_dir, "md5sums.txt")
    with open(md5sum_file, 'w') as fh:
        for fastq_file in fastq_files:
            logger2.debug("Calculating md5 for %s using md5sum" % fastq_file)
            cl = ["md5sum", fastq_file]
            fh.write(subprocess.check_output(cl))


def _clean_qseq(bc_dir, fastq_dir):
    """Remove the temporary qseq files if the corresponding fastq file
       has been created
    """
    glob_str = "*_1_fastq.txt"
    fastq_files = glob.glob(os.path.join(fastq_dir, glob_str))

    for fastq_file in fastq_files:
        try:
            lane = int(os.path.basename(fastq_file)[0])
        except ValueError:
            continue

        logger2.debug("Removing qseq files for lane %d" % lane)
        glob_str = "s_%d_*qseq.txt" % lane

        for qseq_file in glob.glob(os.path.join(bc_dir, glob_str)):
            try:
                os.unlink(qseq_file)
            except:
                logger2.debug("Could not remove %s" % qseq_file)


def _generate_qseq(bc_dir, config):
    """Generate qseq files from illumina bcl files if not present.

    More recent Illumina updates do not produce qseq files. Illumina's
    offline base caller (OLB) generates these starting with bcl,
    intensity and filter files.
    """
    if not os.path.exists(os.path.join(bc_dir, "finished.txt")):
        bcl2qseq_log = os.path.join(config["log_dir"], "setupBclToQseq.log")
        cmd = os.path.join(config["program"]["olb"], "bin", "setupBclToQseq.py")
        cl = [cmd, "-L", bcl2qseq_log, "-o", bc_dir, "--in-place", "--overwrite",
              "--ignore-missing-stats", "--ignore-missing-control"]
        # in OLB version 1.9, the -i flag changed to intensities instead of input
        version_cl = [cmd, "-v"]
        p = subprocess.Popen(version_cl, stdout=subprocess.PIPE)
        (out, _) = p.communicate()
        olb_version = float(out.strip().split()[-1].rsplit(".", 1)[0])
        if olb_version > 1.8:
            cl += ["-P", ".clocs"]
            cl += ["-b", bc_dir]
        else:
            cl += ["-i", bc_dir, "-p", os.path.split(bc_dir)[0]]
        subprocess.check_call(cl)
        with utils.chdir(bc_dir):
            processors = config["algorithm"].get("num_cores", 8)
            cl = config["program"].get("olb_make", "make").split() + ["-j", str(processors)]
            subprocess.check_call(cl)


def _is_finished_dumping(directory):
    """Determine if the sequencing directory has all files.

    The final checkpoint file will differ depending if we are a
    single or paired end run.
    """
    # Check final output files; handles both HiSeq, MiSeq and GAII

    to_check = ["Basecalling_Netcopy_complete_SINGLEREAD.txt",
                "Basecalling_Netcopy_complete_READ2.txt"]

    # Bugfix: On case-isensitive filesystems (e.g. MacOSX), the READ2-check will return true
    # http://stackoverflow.com/questions/6710511/case-sensitive-path-comparison-in-python
    for fname in os.listdir(directory):
        if fname in to_check:
            return True

    return _is_finished_basecalling_read(directory, _expected_reads(directory))


def _is_finished_first_base_report(directory):
    """Determine if the first base report has been generated
    """
    return os.path.exists(os.path.join(directory,
                                       "First_Base_Report.htm"))


def _is_started_initial_processing(directory):
    """Determine if initial processing has been started
    """
    return os.path.exists(os.path.join(directory,
                                       "initial_processing_started.txt"))


def _is_initial_processing(directory):
    """Determine if initial processing is in progress
    """
    return (_is_started_initial_processing(directory) and
            not os.path.exists(os.path.join(directory,
                                            "initial_processing_completed.txt")))


def _is_started_first_read_processing(directory):
    """Determine if processing of first read has been started
    """
    return os.path.exists(os.path.join(directory,
                                       "first_read_processing_started.txt"))


def _is_processing_first_read(directory):
    """Determine if processing of first read is in progress
    """
    return (_is_started_first_read_processing(directory) and
            not os.path.exists(os.path.join(directory,
                                            "first_read_processing_completed.txt")))


def _is_started_second_read_processing(directory):
    """Determine if processing of second read of the pair has been started
    """
    return os.path.exists(os.path.join(directory,
                                       "second_read_processing_started.txt"))


def _is_finished_basecalling_read(directory, readno):
    """
    Determine if a given read has finished being basecalled. Raises a ValueError if
    the run is not configured to produce the read
    """
    if readno < 1 or readno > _expected_reads(directory):
        raise ValueError("The run will not produce a Read{:d}".format(readno))

    return os.path.exists(os.path.join(directory,
                                       "Basecalling_Netcopy_complete_Read{:d}.txt".format(readno)))


def _do_initial_processing(directory):
    """Determine if the initial processing actions should be run
    """
    # A miseq run does not generate a first base report 
    return ((_is_miseq_run(directory) or 
             _is_finished_first_base_report(directory)) and
            not _is_started_initial_processing(directory))


def _do_first_read_processing(directory):
    """Determine if the processing of the first read should be run
    """
    # If run is not indexed, the first read itself is the highest number
    read = max(1, _last_index_read(directory))

    # FIXME: Handle a case where the index reads are the first to be read
    return (_is_finished_basecalling_read(directory, read) and
            not _is_initial_processing(directory) and
            not _is_started_first_read_processing(directory))


def _do_second_read_processing(directory):
    """Determine if the processing of the second read of the pair should be run
    """
    return (_is_finished_dumping(directory) and
            not _is_initial_processing(directory) and
            not _is_processing_first_read(directory) and
            not _is_started_second_read_processing(directory))


def _last_index_read(directory):
    """Parse the number of the highest index read from the RunInfo.xml
    """
    read_numbers = [int(read.get("Number", 0)) for read in _get_read_configuration(directory) if read.get("IsIndexedRead", "") == "Y"]
    return 0 if len(read_numbers) == 0 else max(read_numbers)


def _expected_reads(directory):
    """Parse the number of expected reads from the RunInfo.xml file.
    """
    return len(_get_read_configuration(directory))

def _is_miseq_run(fcdir):
    """Return True if this is a miseq run folder, False otherwise
    """
    if not _is_run_folder_name(os.path.basename(fcdir)):
        return False
    
    # Assume that a HiSeq run folder ends with [AB][A-Z0-9]XX and that it is a MiSeq folder otherwise
    p = os.path.basename(fcdir).split("_")[-1]
    m = re.match(r'[AB][A-Z0-9]+XX',p)
    return (m is None)

def _is_finished_dumping_checkpoint(directory):
    """Recent versions of RTA (1.10 or better), write the complete file.

    This is the most straightforward source but as of 1.10 still does not
    work correctly as the file will be created at the end of Read 1 even
    if there are multiple reads.
    """
    check_file = os.path.join(directory, "Basecalling_Netcopy_complete.txt")
    check_v1, check_v2 = (1, 10)
    if os.path.exists(check_file):
        with open(check_file) as in_handle:
            line = in_handle.readline().strip()
        if line:
            version = line.split()[-1]
            v1, v2 = [float(v) for v in version.split(".")[:2]]
            if ((v1 > check_v1) or (v1 == check_v1 and v2 >= check_v2)):
                return True


def _get_read_configuration(directory):
    """Parse the RunInfo.xml w.r.t. read configuration and return a list of dicts
    """
    reads = []
    run_info_file = os.path.join(directory, "RunInfo.xml")
    if os.path.exists(run_info_file):
        tree = ET.ElementTree()
        tree.parse(run_info_file)
        read_elem = tree.find("Run/Reads")
        for read in read_elem:
            reads.append(dict(zip(read.keys(), [read.get(k) for k in read.keys()])))

    return sorted(reads, key=lambda r: int(r.get("Number", 0)))


def _get_flowcell_id(directory):
    """Parese the RunInfo.xml and return the Flowcell ID
    """
    run_info_file = os.path.join(directory, "RunInfo.xml")
    flowcell_id = ''
    if os.path.exists(run_info_file):
        tree = ET.ElementTree()
        tree.parse(run_info_file)
        flowcell_id = tree.find("Run/Flowcell").text
    return flowcell_id


def _get_bases_mask(directory):
    """Get the base mask to use with Casava based on the run configuration and
    on the run SampleSheet
    """
    runsetup = _get_read_configuration(directory)
    flowcell_id = _get_flowcell_id(directory)
    base_masks = {}

    #Create groups of reads by index length
    ss_name = os.path.join(directory, str(flowcell_id) + '.csv')
    if os.path.exists(ss_name):
        ss = csv.DictReader(open(ss_name, 'rb'), delimiter=',')
        samplesheet = []
        [samplesheet.append(read) for read in ss]
        for r in samplesheet:
            index_length = len(r['Index'].replace('-', ''))
            if not base_masks.has_key(index_length):
                base_masks[index_length] = {'base_mask': [],
                                            'samples': {'fieldnames': ss.fieldnames, 'samples':[]}}
            base_masks[index_length]['samples']['samples'].append(r)

    #Create the basemask for each group
    for index_size, index_group in base_masks.iteritems():
        index_size = index_size
        group = index_size
        bm = []
        for read in runsetup:
            cycles = read['NumCycles']
            if read['IsIndexedRead'] == 'N':
                bm.append('Y' + cycles)
            else:
                if index_size > 0:
                    if index_size < int(cycles):
                        m = 'I' + str(index_size) + 'N'
                        if int(cycles) - index_size > 1:
                            bm.append(m + str(int(cycles) - index_size))
                        else:
                            bm.append(m)
                        index_size = 0
                    elif index_size >= int(cycles):
                        bm.append('I' + cycles)
                        index_size = index_size - int(cycles)
                else:
                    bm.append('N' + cycles)
        base_masks[group]['base_mask'] = bm
    return base_masks


def _files_to_copy(directory):
    """Retrieve files that should be remotely copied.
    """

    #First include the files in the root directory, otherwise
    #the --include=*/ makes a match with all subdirectories
    #with extension txt, csv, err or out
    with utils.chdir(directory):
        root_files = reduce(operator.add,
                            [glob.glob("*.xml"),
                             glob.glob("*.csv"),
                             glob.glob("*.txt"),
                             glob.glob("*.err"),
                             glob.glob("*.out")])

    reports = ["Data/Intensities/BaseCalls/*.xml", \
                "Data/Intensities/BaseCalls/*.xsl", \
                "Data/Intensities/BaseCalls/*.htm", \
                "Unaligned*/Basecall_Stats_*/*", \
                "Unaligned*/Basecall_Stats_*/**/*", \
                "Data/Intensities/BaseCalls/Plots", \
                "Data/reports", \
                "Data/Status.htm", \
                "Data/Status_Files", "InterOp"]

    run_info = ["run_info.yaml", \
                "Unaligned*/Project_*/**/*.csv", \
                "Unaligned*/Undetermined_indices/**/*.csv"]

    fastq = ["Data/Intensities/BaseCalls/*fastq.gz", \
            "Unaligned*/Project_*/**/*.fastq.gz", \
            "Unaligned*/Undetermined_indices/**/*.fastq.gz", \
            "Data/Intensities/BaseCalls/fastq"]

    analysis = ["Data/Intensities/BaseCalls/Alignment"]

    patterns = root_files + reports + run_info + fastq + analysis

    with utils.chdir(directory):
        image_redo_files = reduce(operator.add,
                                  [glob.glob("*.params"),
                                   glob.glob("Images/L*/C*"),
                                   ["RunInfo.xml", "runParameters.xml"]])

        qseqs = reduce(operator.add,
                    [glob.glob("Data/Intensities/*.xml"),
                     glob.glob("Data/Intensities/BaseCalls/*qseq.txt")
                    ])

        reports = reduce(operator.add,
                        [glob.glob("*.xml"),
                         glob.glob("Data/Intensities/BaseCalls/*.xml"),
                         glob.glob("Data/Intensities/BaseCalls/*.xsl"),
                         glob.glob("Data/Intensities/BaseCalls/*.htm"),
                         glob.glob("Unaligned*/Basecall_Stats_*/*"),
                         glob.glob("Unaligned*/Basecall_Stats_*/**/*"),
                         ["Data/Intensities/BaseCalls/Plots", "Data/reports",
                          "Data/Status.htm", "Data/Status_Files", "InterOp"]
                        ])

        run_info = reduce(operator.add,
                        [glob.glob("run_info.yaml"),
                         glob.glob("*.csv"),
                         glob.glob("Unaligned*/Project_*/**/*.csv"),
                         glob.glob("Unaligned*/Undetermined_indices/**/*.csv"),
                         glob.glob("*.txt"),
                         glob.glob("*.err"),
                         glob.glob("*.out"),
                        ])

        logs = reduce(operator.add, [["Logs", "Recipe", "Diag", "Data/RTALogs", "Data/Log.txt"]])


    #For process_files return a list of patterns to be included in the rsync command
    #instead of including the list of corresponding files. Otherwise the rsync
    #command becomes too long
    return (sorted(image_redo_files + logs + reports + run_info + qseqs),
            patterns, ["*"])


def _read_reported(msg_db):
    """Retrieve a list of directories previous reported.
    """
    reported = []
    if os.path.exists(msg_db):
        with open(msg_db) as in_handle:
            for line in in_handle:
                reported.append(line.strip())
    else:
        # Just check if the path to the file exists
        utils.safe_makedir(os.path.dirname(msg_db))
        open(msg_db, 'w')
    return reported

def _is_run_folder_name(name):
    """Check if a name matches the format of *iSeq run folders"""
    m = re.match("\d{6}_[A-Za-z0-9]+_\d+_[AB]?[A-Z0-9\-]+", name)
    return (m is not None)

def _get_directories(config):
    for directory in config["dump_directories"]:
        for fpath in sorted(os.listdir(directory)):
            if not os.path.isdir(os.path.join(directory, fpath)) or not _is_run_folder_name(fpath):
                continue
            yield os.path.join(directory, fpath)


def _update_reported(msg_db, new_dname):
    """Add a new directory to the database of reported messages.
    """
    reported = _read_reported(msg_db)
    for d in [dir for dir in reported if dir.startswith(new_dname)]:
        new_dname = d
        reported.remove(d)
    reported.append("%s\t%s" % (new_dname, time.strftime("%x-%X")))

    with open(msg_db, "w") as out_handle:
        for dir in reported:
            out_handle.write("%s\n" % dir)


def finished_message(fn_name, run_module, directory, files_to_copy,
                     config, config_file, pushed=False):
    """Wait for messages with the give tag, passing on to the supplied handler.
    """
    logger2.debug("Calling remote function: %s" % fn_name)
    user = getpass.getuser()
    hostname = socket.gethostbyaddr(socket.gethostname())[0]
    data = dict(
            machine_type='illumina',
            hostname=hostname,
            user=user,
            directory=directory,
            to_copy=files_to_copy
            )
    dirs = {"work": os.getcwd(),
            "config": os.path.dirname(config_file)}

    runner = messaging.runner(run_module, dirs, config, config_file, wait=False)

    if pushed:
        config["directory"] = directory
        runner(fn_name, [[config]])

    else:
        runner(fn_name, [[data]])

if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-b", "--backup", dest="backup_msg",
            action="store_true", default=False)
    parser.add_option("-d", "--fetch", dest="fetch_msg",
            action="store_true", default=False)
    parser.add_option("-p", "--process", dest="process_msg",
            action="store_true", default=False)
    parser.add_option("-s", "--store", dest="store_msg",
            action="store_true", default=False)
    parser.add_option("-f", "--fastq", dest="fastq",
            action="store_true", default=False)
    parser.add_option("-z", "--compress-fastq", dest="compress_fastq",
            action="store_true", default=False)
    parser.add_option("-q", "--qseq", dest="qseq",
            action="store_true", default=False)
    parser.add_option("-c", "--pre-casava", dest="casava",
            action="store_false", default=True)
    parser.add_option("-r", "--remove-qseq", dest="remove_qseq",
            action="store_true", default=False)
    parser.add_option("-m", "--miseq", dest="miseq",
            action="store_true", default=False)
    parser.add_option("--pull_data", dest="push_data",
            action="store_false", default=True)
    parser.add_option("--post-process-only", dest="post_process_only",
            action="store_true", default=False)
    parser.add_option("--run-id", dest="run_id",
            action="store", default=None)
    parser.add_option("--no-casava-processing", dest="no_casava_processing",
            action="store_true", default=False)

    (options, args) = parser.parse_args()

    # Option --miseq implies --noprocess, --nostore, --nofastq, --noqseq
    if options.miseq:
        options.fetch_msg = False
        options.process_msg = False
        options.store_msg = False
        options.backup_msg = True
        options.fastq = False
        options.qseq = False
        options.casava = False

    kwargs = {"fetch_msg": options.fetch_msg, \
              "process_msg": options.process_msg, \
              "store_msg": options.store_msg, \
              "backup_msg": options.backup_msg, \
              "fastq": options.fastq, \
              "qseq": options.qseq, \
              "remove_qseq": options.remove_qseq, \
              "compress_fastq": options.compress_fastq, \
              "casava": options.casava, \
              "push_data": options.push_data, \
              "post_process_only": options.post_process_only, \
              "run_id": options.run_id, \
              "no_casava_processing": options.no_casava_processing}

    main(*args, **kwargs)


### Tests ###

import unittest
import shutil
import tempfile


class TestCallsTo_post_process_run(unittest.TestCase):
    def setUp(self):
        self.kwargs = {"fetch_msg": None, \
                       "process_msg": None, \
                       "store_msg": None, \
                       "backup_msg": None, \
                       "fastq": None, \
                       "qseq": None, \
                       "remove_qseq": None, \
                       "compress_fastq": None, \
                       "casava": None, \
                       "post_process_config": None}

    def test_call_in_initial_processing(self):
        args = ["", None, ""]  # [dname, config, local_config]
        self.assertRaises(ValueError, initial_processing, *args, **self.kwargs)

    def test_call_as_in_process_first_read(self):
        args = ["", None, "", ""]  # [dname, config, local_config, unaligned_dir]
        self.assertRaises(OSError, _post_process_run, *args, **self.kwargs)


class TestCheckpoints(unittest.TestCase):
    def setUp(self):
        self.rootdir = tempfile.mkdtemp(prefix="ifm_test_checkpoints_", dir=self.basedir)

    def tearDown(self):
        shutil.rmtree(self.rootdir)

    @classmethod
    def _runinfo(cls, outfile, bases_mask="Y101,I7,Y101"):
        """Return an xml string representing the contents of a RunInfo.xml
        file with the specified read configuration
        """
        root = ET.Element("RunInfo")
        run = ET.Element("Run", attrib={"Id": "120924_SN0002_0003_CC003CCCXX",
                                        "Number": "1"})
        root.append(run)
        run.append(ET.Element("Flowcell", text="C003CCCXX"))
        run.append(ET.Element("Instrument", text="SN0002"))
        run.append(ET.Element("Date", text="120924"))

        reads = ET.Element("Reads")
        for n, r in enumerate(bases_mask.split(",")):
            reads.append(ET.Element("Read", attrib={"Number": str(n + 1),
                                                    "NumCycles": r[1:],
                                                    "IsIndexedRead": "Y" if r[0] == "I" else "N"}))
        run.append(reads)
        run.append(ET.Element("FlowcellLayout", attrib={"LaneCount": "8",
                                                        "SurfaceCount": "2",
                                                        "SwathCount": "3",
                                                        "TileCount": "16"}))

        et = ET.ElementTree(root)
        et.write(outfile,encoding="UTF-8")
        return outfile


    @classmethod
    def _samplesinfo(cls, outfile, index_info='simple_index'):
        """Return a csv string representing the contents of a samples csv file
        with the specified index configuration.
        """
        fn = ['FCID', 'Lane', 'SampleID', 'SampleRef', 'Index', 'Description', \
              'Control', 'Recipe', 'Operator', 'SampleProject']
        sample = "C1NWWACXX,1,P352_184B_index12,hg19,{index},J_Doe_13_01,N,R1,NN,J_Doe_13_01"
        with open(outfile, 'w') as samplesheet:
            ss = csv.DictWriter(samplesheet, fieldnames=fn, dialect='excel')
            ss.writeheader()
            #Write samples according to index configuration
            rows = []
            if index_info == 'simple_index':
                s1 = sample.format(index='ACGTAG').split(',')
                rows = [s1]
            elif index_info == 'mixed_index':
                s1 = sample.format(index='ACGTAG').split(',')
                s2 = sample.format(index='ACGTACGT').split(',')
                s3 = sample.format(index='ACGTACGT-ACGTACGT').split(',')
                rows = [s1, s2, s3]
            elif index_info == 'no_index':
                s1 = sample.format(index='').split(',')
                rows = [s1]
            ss.writerows([{k:v for k,v in izip(fn,r)} for r in rows])
        return outfile


    @classmethod
    def setUpClass(cls):
        cls.basedir = tempfile.mkdtemp(prefix="ifm_test_checkpoints_base_")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.basedir)

    def test__is_finished_first_base_report(self):
        """First base report"""
        self.assertFalse(_is_finished_first_base_report(self.rootdir))
        utils.touch_file(os.path.join(self.rootdir,"First_Base_Report.htm"))
        self.assertTrue(_is_finished_first_base_report(self.rootdir))

    def test__is_started_initial_processing(self):
        """Initial processing started"""
        self.assertFalse(_is_started_initial_processing(self.rootdir))
        utils.touch_indicator_file(os.path.join(self.rootdir,"initial_processing_started.txt"))
        self.assertTrue(_is_started_initial_processing(self.rootdir))

    def test__is_started_first_read_processing(self):
        """First read processing started
        """
        self.assertFalse(_is_started_first_read_processing(self.rootdir))
        utils.touch_indicator_file(os.path.join(self.rootdir,"first_read_processing_started.txt"))
        self.assertTrue(_is_started_first_read_processing(self.rootdir))

    def test__is_started_second_read_processing(self):
        """Second read processing started
        """
        self.assertFalse(_is_started_second_read_processing(self.rootdir))
        utils.touch_indicator_file(os.path.join(self.rootdir,"second_read_processing_started.txt"))
        self.assertTrue(_is_started_second_read_processing(self.rootdir))

    def test__is_initial_processing(self):
        """Initial processing in progress"""
        self.assertFalse(_is_initial_processing(self.rootdir),
                         "No indicator files should not indicate processing in progress")
        utils.touch_indicator_file(os.path.join(self.rootdir,"initial_processing_started.txt"))
        self.assertTrue(_is_initial_processing(self.rootdir),
                        "Started indicator file should indicate processing in progress")
        utils.touch_indicator_file(os.path.join(self.rootdir,"initial_processing_completed.txt"))
        self.assertFalse(_is_initial_processing(self.rootdir),
                        "Completed indicator file should not indicate processing in progress")

    def test__is_processing_first_read(self):
        """First read processing in progress
        """
        self.assertFalse(_is_processing_first_read(self.rootdir),
                         "No indicator files should not indicate processing in progress")
        utils.touch_indicator_file(os.path.join(self.rootdir,"first_read_processing_started.txt"))
        self.assertTrue(_is_processing_first_read(self.rootdir),
                        "Started indicator file should indicate processing in progress")
        utils.touch_indicator_file(os.path.join(self.rootdir,"first_read_processing_completed.txt"))
        self.assertFalse(_is_processing_first_read(self.rootdir),
                        "Completed indicator file should not indicate processing in progress")

    def test__do_initial_processing(self):
        """Initial processing logic
        """
        self.assertFalse(_do_initial_processing(self.rootdir),
                         "Initial processing should not be run with missing indicator flags")
        utils.touch_file(os.path.join(self.rootdir,"First_Base_Report.htm"))
        self.assertTrue(_do_initial_processing(self.rootdir),
                         "Initial processing should be run after first base report creation")
        utils.touch_indicator_file(os.path.join(self.rootdir,"initial_processing_started.txt"))
        self.assertFalse(_do_initial_processing(self.rootdir),
                         "Initial processing should not be run when processing has been started")
        os.unlink(os.path.join(self.rootdir,"First_Base_Report.htm"))
        self.assertFalse(_do_initial_processing(self.rootdir),
                         "Initial processing should not be run when processing has been started " \
                         "and missing first base report")

    def test__do_first_read_processing(self):
        """First read processing logic
        """
        runinfo = os.path.join(self.rootdir, "RunInfo.xml")
        self._runinfo(runinfo)
        self.assertFalse(_do_first_read_processing(self.rootdir),
                         "Processing should not be run before first read is finished")
        utils.touch_file(os.path.join(self.rootdir,
                                      "Basecalling_Netcopy_complete_Read1.txt"))
        self.assertFalse(_do_first_read_processing(self.rootdir),
                         "Processing should not be run before last index read is finished")
        utils.touch_file(os.path.join(self.rootdir,
                                      "Basecalling_Netcopy_complete_Read2.txt"))
        utils.touch_indicator_file(os.path.join(self.rootdir,
                                                "initial_processing_started.txt"))
        self.assertFalse(_do_first_read_processing(self.rootdir),
                         "Processing should not be run when previous processing step is in progress")
        utils.touch_indicator_file(os.path.join(self.rootdir,
                                                "initial_processing_completed.txt"))
        self.assertTrue(_do_first_read_processing(self.rootdir),
                        "Processing should be run when last index read is finished")
        utils.touch_indicator_file(os.path.join(self.rootdir,
                                                "first_read_processing_started.txt"))
        self.assertFalse(_do_first_read_processing(self.rootdir),
                         "Processing should not be run when processing has started")

    def test__do_second_read_processing(self):
        """Second read processing logic
        """
        runinfo = os.path.join(self.rootdir, "RunInfo.xml")
        self._runinfo(runinfo)
        utils.touch_file(os.path.join(self.rootdir,
                                      "Basecalling_Netcopy_complete_READ2.txt"))
        self.assertTrue(_do_second_read_processing(self.rootdir),
                        "Processing should be run when last read GAII checkpoint exists")
        os.unlink(os.path.join(self.rootdir,
                               "Basecalling_Netcopy_complete_READ2.txt"))
        self.assertFalse(_do_second_read_processing(self.rootdir),
                         "Processing should not be run before any reads are finished")
        utils.touch_file(os.path.join(self.rootdir,
                                      "Basecalling_Netcopy_complete_Read2.txt"))
        self.assertFalse(_do_second_read_processing(self.rootdir),
                         "Processing should not be run before last read is finished")
        utils.touch_file(os.path.join(self.rootdir,
                                      "Basecalling_Netcopy_complete_Read3.txt"))
        self.assertTrue(_do_second_read_processing(self.rootdir),
                        "Processing should be run when last read is finished")
        utils.touch_indicator_file(os.path.join(self.rootdir,
                                                "second_read_processing_started.txt"))
        self.assertFalse(_do_second_read_processing(self.rootdir),
                         "Processing should not be run when processing has started")

    def test__expected_reads(self):
        """Get expected number of reads
        """
        self.assertEqual(_expected_reads(self.rootdir),0,
                         "Non-existant RunInfo.xml should return 0 expected reads")

        runinfo = os.path.join(self.rootdir,"RunInfo.xml")
        self._runinfo(runinfo)
        self.assertEqual(_expected_reads(self.rootdir),3,
                         "Default RunInfo.xml should return 3 expected reads")

        self._runinfo(runinfo, "Y101,I6,I6,Y101")

        self.assertEqual(_expected_reads(self.rootdir),4,
                         "Dual-index RunInfo.xml should return 4 expected reads")

    def test__last_index_read(self):
        """Get number of last index read
        """
        self.assertEqual(_last_index_read(self.rootdir),0,
                         "Non-existant RunInfo.xml should return 0 as last index read")

        runinfo = os.path.join(self.rootdir,"RunInfo.xml")
        self._runinfo(runinfo)
        self.assertEqual(_last_index_read(self.rootdir),2,
                         "Default RunInfo.xml should return 2 as last index read")

        self._runinfo(runinfo, "Y101,I6,I6,Y101")
        self.assertEqual(_last_index_read(self.rootdir),3,
                         "Dual-index RunInfo.xml should return 3 as last expected read")

        self._runinfo(runinfo, "Y101,Y101,Y101")
        self.assertEqual(_last_index_read(self.rootdir),0,
                         "Non-index RunInfo.xml should return 0 as last expected read")

    def test__is_finished_basecalling_read(self):
        """Detect finished read basecalling
        """

        # Create a custom RunInfo.xml in the current directory
        runinfo = os.path.join(self.rootdir,"RunInfo.xml")
        self._runinfo(runinfo, "Y101,Y101")

        with self.assertRaises(ValueError):
            _is_finished_basecalling_read(self.rootdir,0)

        with self.assertRaises(ValueError):
            _is_finished_basecalling_read(self.rootdir,3)

        for read in (1,2):
            self.assertFalse(_is_finished_basecalling_read(self.rootdir,read),
                             "Should not return true with missing indicator file")
            utils.touch_file(os.path.join(self.rootdir,
                                          "Basecalling_Netcopy_complete_Read{:d}.txt".format(read)))
            self.assertTrue(_is_finished_basecalling_read(self.rootdir,read),
                            "Should return true with existing indicator file")

    def test__get_bases_mask(self):
        """Get bases mask
        """
        runinfo = os.path.join(self.rootdir, "RunInfo.xml")
        self._runinfo(runinfo)
        flowcell_id = _get_flowcell_id(self.rootdir)
        samplesinfo = os.path.join(self.rootdir, str(flowcell_id) + '.csv')
        #Test simple index
        self._samplesinfo(samplesinfo)
        self.assertEqual(_get_bases_mask(self.rootdir)[6]['base_mask'], ['Y101', 'I6N', 'Y101'])
        #Test mixed indexes
        self._runinfo(runinfo, bases_mask='Y101,I8,I8,Y101')
        self._samplesinfo(samplesinfo, index_info='mixed_index')
        masks = _get_bases_mask(self.rootdir)
        self.assertEqual(masks[6]['base_mask'], ['Y101', 'I6N2', 'N8', 'Y101'])
        self.assertEqual(masks[8]['base_mask'], ['Y101', 'I8', 'N8', 'Y101'])
        self.assertEqual(masks[16]['base_mask'], ['Y101', 'I8', 'I8', 'Y101'])
        #Test no index
        self._runinfo(runinfo, bases_mask='Y101,Y101')
        self._samplesinfo(samplesinfo, index_info='no_index')
        self.assertEqual(_get_bases_mask(self.rootdir)[0]['base_mask'], ['Y101', 'Y101'])




    def test__get_read_configuration(self):
        """Get read configuration
        """

        self.assertListEqual(_get_read_configuration(self.rootdir), [],
                             "Expected empty list for non-existing RunInfo.xml")

        runinfo = os.path.join(self.rootdir,"RunInfo.xml")
        self._runinfo(runinfo)
        obs_reads = _get_read_configuration(self.rootdir)
        self.assertListEqual([r.get("Number",0) for r in obs_reads],["1","2","3"],
                             "Expected 3 reads for 2x100 PE")

    def test__get_directories(self):
        """Get run output directories
        """
        config = {"dump_directories": [self.rootdir]}
        obs_dirs = [d for d in _get_directories(config)]
        self.assertListEqual([],obs_dirs,
                              "Expected empty list for getting non-existing run directories")
        utils.touch_file(os.path.join(self.rootdir, "111111_SN111_1111_A11111111"))
        obs_dirs = [d for d in _get_directories(config)]
        self.assertListEqual([],obs_dirs,
                              "Should not pick up files, only directories")
        exp_dirs = [os.path.join(self.rootdir, "222222_SN222_2222_A2222222"),
                    os.path.join(self.rootdir, "333333_D0023_3333_B33333XX")]
        os.mkdir(exp_dirs[-1])
        os.mkdir(exp_dirs[-2])
        obs_dirs = [d for d in _get_directories(config)]
        self.assertListEqual(sorted(exp_dirs),sorted(obs_dirs),
                              "Should pick up matching directory - hiseq-style")
        exp_dirs.append(os.path.join(self.rootdir, "333333_M33333_3333_A000000000-A3333"))
        os.mkdir(exp_dirs[-1])
        obs_dirs = [d for d in _get_directories(config)]
        self.assertListEqual(sorted(exp_dirs),sorted(obs_dirs),
                              "Should pick up matching directory - miseq-style")
