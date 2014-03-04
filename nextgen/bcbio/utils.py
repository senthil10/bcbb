"""Helpful utilities for building analysis pipelines.
"""
import os
import stat
import tempfile
import shutil
import contextlib
import itertools
import functools
import ConfigParser
import csv
import codecs
import cStringIO
import gzip
import glob
import re
from datetime import datetime
from xml.etree import ElementTree as ET
from itertools import izip_longest
from bs4 import BeautifulSoup

try:
    import multiprocessing
    from multiprocessing.pool import IMapIterator
except ImportError:
    multiprocessing = None

import yaml


@contextlib.contextmanager
def cpmap(cores=1):
    """Configurable parallel map context manager.

    Returns appropriate map compatible function based on configuration:
    - Local single core (the default)
    - Multiple local cores
    """
    if int(cores) == 1:
        yield itertools.imap
    else:
        if multiprocessing is None:
            raise ImportError("multiprocessing not available")

        # Fix to allow keyboard interrupts in multiprocessing: https://gist.github.com/626518
        def wrapper(func):
            def wrap(self, timeout=None):
                return func(self, timeout=timeout if timeout is not None else 1e100)
            return wrap
        IMapIterator.next = wrapper(IMapIterator.next)
        # recycle threads on Python 2.7; remain compatible with Python 2.6
        try:
            pool = multiprocessing.Pool(int(cores), maxtasksperchild=1)
        except TypeError:
            pool = multiprocessing.Pool(int(cores))

        yield pool.imap_unordered
        pool.terminate()


def map_wrap(f):
    """Wrap standard function to easily pass into 'map' processing.
    """
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        return apply(f, *args, **kwargs)
    return wrapper


def memoize_outfile(ext):
    """Creates outfile from input file and ext, running if outfile not present.

    This requires a standard function usage. The first arg, or kwarg 'in_file', needs
    to be the input file that is being processed. The output name is created with the
    provided ext relative to the input. The function is only run if the created
    out_file is not present.
    """
    def decor(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            if len(args) > 0:
                in_file = args[0]
            else:
                in_file = kwargs['in_file']
            out_file = "%s%s" % (os.path.splitext(in_file)[0], ext)
            if not os.path.exists(out_file) or os.path.getsize(out_file) == 0:
                kwargs['out_file'] = out_file
                f(*args, **kwargs)
            return out_file
        return wrapper
    return decor

def safe_makedir(dname):
    """Make a directory if it doesn't exist, handling concurrent race conditions.
    """
    if not os.path.exists(dname):
        # we could get an error here if multiple processes are creating
        # the directory at the same time. Grr, concurrency.
        try:
            os.makedirs(dname)
        except OSError:
            if not os.path.isdir(dname):
                raise
    return dname


@contextlib.contextmanager
def curdir_tmpdir(remove=True):
    """Context manager to create and remove a temporary directory.
    """
    tmp_dir_base = os.path.join(os.getcwd(), "tmp")
    safe_makedir(tmp_dir_base)
    tmp_dir = tempfile.mkdtemp(dir=tmp_dir_base)
    safe_makedir(tmp_dir)
    # Explicitly change the permissions on the temp directory to make it writable by group
    os.chmod(tmp_dir, stat.S_IRWXU | stat.S_IRWXG)
    try:
        yield tmp_dir
    finally:
        if remove:
            shutil.rmtree(tmp_dir)


@contextlib.contextmanager
def chdir(new_dir):
    """Context manager to temporarily change to a new directory.

    http://lucentbeing.com/blog/context-managers-and-the-with-statement-in-python/
    """
    cur_dir = os.getcwd()
    safe_makedir(new_dir)
    os.chdir(new_dir)
    try:
        yield
    finally:
        os.chdir(cur_dir)


@contextlib.contextmanager
def tmpfile(*args, **kwargs):
    """Make a tempfile, safely cleaning up file descriptors on completion.
    """
    (fd, fname) = tempfile.mkstemp(*args, **kwargs)
    try:
        yield fname
    finally:
        os.close(fd)
        if os.path.exists(fname):
            os.remove(fname)


def file_exists(fname):
    """Check if a file exists and is non-empty.
    """
    return os.path.exists(fname) and os.path.getsize(fname) > 0


def touch_file(fname):
    """Create an empty file
    """
    open(fname, "w").close()

def create_dirs(config, names=None):
    if names is None:
        names = config["dir"].keys()

    for dname in names:
        d = config["dir"][dname]
        safe_makedir(d)


def save_diskspace(fname, reason, config):
    """Overwrite a file in place with a short message to save disk.

    This keeps files as a sanity check on processes working, but saves
    disk by replacing them with a short message.
    """
    if not config["algorithm"].get("save_diskspace", False):
        return

    with open(fname, "w") as out_handle:
        out_handle.write("File removed to save disk space: {}".format(reason))


def read_galaxy_amqp_config(galaxy_config, base_dir):
    """Read connection information on the RabbitMQ server from Galaxy config.
    """
    galaxy_config = add_full_path(galaxy_config, base_dir)
    config = ConfigParser.ConfigParser()
    config.read(galaxy_config)
    amqp_config = {}
    for option in config.options("galaxy_amqp"):
        amqp_config[option] = config.get("galaxy_amqp", option)

    return amqp_config


def add_full_path(dirname, basedir=None):
    if basedir is None:
        basedir = os.getcwd()

    if not dirname.startswith("/"):
        dirname = os.path.join(basedir, dirname)

    return dirname


def compress_files(to_compress):
    """Compress all the files in the set to_compress
    """
    #This local import prevents a circular import, since since transaction import
    #utils, and then utils imports transaction
    from bcbio.distributed.transaction import file_transaction
    raw_size = 0
    gzipped_size = 0
    for f in to_compress:
        out_file = "{}{}".format(str(f),'.gz')
        if file_exists(str(f)) and not file_exists(out_file):
            with file_transaction(out_file) as tx_out_file:
                raw_size += os.stat(f).st_size
                with open(f, 'rb') as f_in:
                    with gzip.open(tx_out_file, 'wb') as f_out:
                        f_out.writelines(f_in)
                os.remove(f)
                gzipped_size += os.stat(tx_out_file).st_size
    return [[raw_size, gzipped_size]]


# ## Dealing with configuration files

def merge_config_files(fnames):
    """Merge configuration files, preferring definitions in latter files.
    """
    def _load_yaml(fname):
        with open(fname) as in_handle:
            config = yaml.load(in_handle)

        return config

    out = _load_yaml(fnames[0])
    for fname in fnames[1:]:
        cur = _load_yaml(fname)
        for k, v in cur.iteritems():
            if k in out and isinstance(out[k], dict):
                out[k].update(v)
            else:
                out[k] = v

    return out

def utc_time():
    """
    Make an utc_time with appended 'Z'
    Borrowed from scilifelab.utils.timestamp
    """
    return str(datetime.utcnow()) + 'Z'


def touch_indicator_file(fname, force=False):
    """Write the current timestamp to the specified file. If it exists, append
    the timestamp to the end
    """
    mode = "w"
    if file_exists(fname) and not force:
        mode = "a"
    with open(fname, mode) as out_handle:
        out_handle.write("{}\n".format(utc_time()))
    return fname


def get_post_process_yaml(self):
    std = os.path.join(self.data_dir, "post_process.yaml")
    sample = os.path.join(self.data_dir, "post_process-sample.yaml")
    if os.path.exists(std):
        return std
    else:
        return sample


def merge_flowcell_demux_summary(u1, u2, fc_id):
    """Merge two Flowcell_Demux_Summary.xml files.

    It assumes the structure:
    <Summary>
        <Lane index="X">
           .
           .
           .
        </Lane index="X">
    </Summary>

    Where X is the lane number [1-8].

    Also assumes that indexes of different length are run in different lanes.

    :param: u1: Unaligned directory where to find the fist file
    :patam: u2: Unaligned directory where to find the second file
    :param: fc_id: Flowcell id

    :return: merged: ElementTree resulting of merging both files.
    """
    #Read the XML to merge
    fc1_f = os.path.join(u1, 'Basecall_Stats_{fc_id}'.format(fc_id=fc_id),
            'Flowcell_demux_summary.xml')
    fc2_f = os.path.join(u2, 'Basecall_Stats_{fc_id}'.format(fc_id=fc_id),
            'Flowcell_demux_summary.xml')
    fc1 = ET.parse(fc1_f).getroot()
    fc2 = ET.parse(fc2_f).getroot()

    #Create a new one and merge there
    merged = ET.ElementTree(ET.Element('Summary'))
    merged_r = merged.getroot()
    lanes = merged_r.getchildren()
    for l1, l2 in izip_longest(fc1.getchildren(), fc2.getchildren()):
        lanes.append(l1) if l1 is not None else []
        lanes.append(l2) if l2 is not None else []

    #Sort the children by lane number and return the merged file
    lanes.sort(key= lambda x: x.attrib['index'])
    return merged


def merge_demultiplex_stats(u1, u2, fc_id):
    """Merge two Demultiplex_Stats.htm files.

    Will append to the Demultiplex_Stats.htm file in u1 the Barcode Lane
    Statistics and Sample Information found in Demultiplex_Stats.htm file in u2.

    The htm file should be structured in such a way that it has two tables (in
    this order): Barcode Lane Statistics and Sample Information. The tables have
    an attribute 'id' which value is ScrollableTableBodyDiv.

    :param: u1: Unaligned directory where to find the fist file
    :patam: u2: Unaligned directory where to find the second file
    :param: fc_id: Flowcell id

    :return: merged: BeautifulSoup object representing the merging of both files.
    """
    with open(os.path.join(u1, 'Basecall_Stats_{fc_id}'.format(fc_id=fc_id),
            'Demultiplex_Stats.htm')) as f:
        ds1 = BeautifulSoup(f.read())
    with open(os.path.join(u2, 'Basecall_Stats_{fc_id}'.format(fc_id=fc_id),
            'Demultiplex_Stats.htm')) as f:
        ds2 = BeautifulSoup(f.read())

    #Get the information from the HTML files
    barcode_lane_statistics_u1, sample_information_u1 = ds1.find_all('div',
        attrs={'id':'ScrollableTableBodyDiv'})
    barcode_lane_statistics_u2, sample_information_u2 = ds2.find_all('div',
        attrs={'id':'ScrollableTableBodyDiv'})

    #Append to the end (tr is the HTML tag under the <div> tag that delimites
    #the sample and barcode statistics information)
    for sample in barcode_lane_statistics_u1.find_all('tr'):
        last_sample = sample
    [last_sample.append(new_sample) for new_sample in \
        barcode_lane_statistics_u2.find_all('tr')]

    for sample in sample_information_u1.find_all('tr'):
        last_sample = sample
    [last_sample.append(new_sample) for new_sample in \
        sample_information_u2.find_all('tr')]

    return ds1


def merge_undemultiplexed_stats_metrics(u1, u2, fc_id):
    """Merge and sort two Undemultiplexed_stats.metrics files.
    """
    with open(os.path.join(u1, 'Basecall_Stats_{fc_id}'.format(fc_id=fc_id),
            'Undemultiplexed_stats.metrics'), 'a+') as us1:
        with open(os.path.join(u2, 'Basecall_Stats_{fc_id}'.format(fc_id=fc_id),
                'Undemultiplexed_stats.metrics')) as us2:
            header = us1.readline()
            lines = []
            for line in us1.readlines():
                lines.append(line.split())
            for line in us2.readlines()[1:]:
                lines.append(line.split())

            # Sort first by index count in descending order, then by read number
            lines = [[line[0], line[1], int(line[2])] for line in lines]
            lines = sorted(lines, key = lambda count: count[2], reverse=True)
            lines = sorted(lines, key = lambda read: read[0])

            us1.seek(0)
            us1.truncate()
            us1.writelines(header)
            for line in lines:
                us1.writelines("\t".join(str(line_field) for line_field in line) + "\n")
            

def merge_demux_results(fc_dir):
    """Merge results of demultiplexing from different Unaligned_Xbp folders
    """
    unaligned_dirs = glob.glob(os.path.join(fc_dir, 'Unaligned_*'))
    #If it is a MiSeq run, the fc_id will be everything after the -
    if '-' in os.path.basename(fc_dir):
        fc_id = os.path.basename(fc_dir).split('_')[-1]
    #If it is a HiSeq run, we only want the flowcell id (without A/B)
    else:
        fc_id = os.path.basename(fc_dir).split('_')[-1][1:]
    basecall_dir = 'Basecall_Stats_{fc_id}'.format(fc_id=fc_id)
    merged_dir = os.path.join(fc_dir, 'Unaligned')
    merged_basecall_dir = os.path.join(merged_dir, basecall_dir)
    #Create the final Unaligned folder and copy there all configuration files
    safe_makedir(os.path.join(merged_dir, basecall_dir))
    shutil.copy(os.path.join(unaligned_dirs[0], basecall_dir,
                    'Flowcell_demux_summary.xml'), merged_basecall_dir)
    shutil.copy(os.path.join(unaligned_dirs[0], basecall_dir,
                    'Demultiplex_Stats.htm'), merged_basecall_dir)
    #The file Undemultiplexed_stats.metrics may not always be there.
    u_s_file = os.path.exists(os.path.join(unaligned_dirs[0], basecall_dir,
                            'Undemultiplexed_stats.metrics'))
    if u_s_file:
        shutil.copy(os.path.join(unaligned_dirs[0], basecall_dir,
                    'Undemultiplexed_stats.metrics'), merged_basecall_dir)
        #And it is possible that it is empty, in which case we have to add
        #the header
        u_s_file_final = os.path.join(merged_basecall_dir, 'Undemultiplexed_stats.metrics')
        with open(u_s_file_final, 'r') as f:
            content = f.readlines()
            header = ['lane', 'sequence', 'count', 'index_name']
            if content and content[0].split() != header:
                with open(u_s_file_final, 'w') as final:
                    final.writelines('\t'.join(header) + '\n')
    if len(unaligned_dirs) > 1:
        for u in unaligned_dirs[1:]:
            #Merge Flowcell_demux_summary.xml
            m_flowcell_demux = merge_flowcell_demux_summary(merged_dir, u, fc_id)
            m_flowcell_demux.write(os.path.join(merged_dir, basecall_dir,
                            'Flowcell_demux_summary.xml'))

            #Merge Demultiplex_Stats.htm
            m_demultiplex_stats = merge_demultiplex_stats(merged_dir, u, fc_id)
            with open(os.path.join(merged_dir, basecall_dir, 'Demultiplex_Stats.htm'), 'w+') as f:
                f.writelines(re.sub(r"Unaligned_[0-9]{1,2}bp", 'Unaligned',
                    m_demultiplex_stats.renderContents()))

            #Merge Undemultiplexed_stats.metrics
            if u_s_file:
                merge_undemultiplexed_stats_metrics(merged_dir, u, fc_id)

# UTF-8 methods for csv module (does not support it in python >2.7)
# http://docs.python.org/library/csv.html#examples


class UTF8Recoder:
    """Iterator that reads an encoded stream and reencodes the input to UTF-8
    """
    def __init__(self, f, encoding):
        self.reader = codecs.getreader(encoding)(f)

    def __iter__(self):
        return self

    def next(self):
        return self.reader.next().encode("utf-8")


class UnicodeReader:
    """A CSV reader which will iterate over lines in the CSV file "f",
       which is encoded in the given encoding.
    """
    def __init__(self, f, dialect=csv.excel, encoding="utf-8", **kwds):
        f = UTF8Recoder(f, encoding)
        self.reader = csv.reader(f, dialect=dialect, **kwds)

    def next(self):
        row = self.reader.next()
        return [unicode(s, "utf-8") for s in row]

    def __iter__(self):
        return self


class UnicodeWriter:
    """A CSV writer which will write rows to CSV file "f",
       which is encoded in the given encoding.
    """

    def __init__(self, f, dialect=csv.excel, encoding="utf-8", **kwds):
        # Redirect output to a queue
        self.queue = cStringIO.StringIO()
        self.writer = csv.writer(self.queue, dialect=dialect, **kwds)
        self.stream = f
        self.encoder = codecs.getincrementalencoder(encoding)()

    def writerow(self, row):
        self.writer.writerow([str(s).encode("utf-8") for s in row])
        # Fetch UTF-8 output from the queue ...
        data = self.queue.getvalue()
        data = data.decode("utf-8")
        # ... and reencode it into the target encoding
        data = self.encoder.encode(data)
        # write to the target stream
        self.stream.write(data)
        # empty queue
        self.queue.truncate(0)

    def writerows(self, rows):
        for row in rows:
            self.writerow(row)

class RecordProgress:
    """A simple interface for recording progress of the parallell
       workflow and outputting timestamp files
    """
    def __init__(self, work_dir, force_overwrite=False):
        self.step = 0
        self.dir = work_dir
        self.fo = force_overwrite

    def progress(self, action):
        self.step += 1
        self._timestamp_file(action)

    def _action_fname(self, action):
        return os.path.join(self.dir, "{s:02d}_{act}.txt".format(s=self.step,act=action))

    def _timestamp_file(self, action):
        """Write a timestamp to the specified file, either appending or
        overwriting an existing file
        """
        fname = self._action_fname(action)
        touch_indicator_file(fname,self.fo)


