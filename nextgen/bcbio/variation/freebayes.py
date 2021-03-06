"""Bayesian variant calling with FreeBayes.

http://bioinformatics.bc.edu/marthlab/FreeBayes
"""
import os
import shutil
import subprocess

from bcbio.utils import file_exists
from bcbio.distributed.transaction import file_transaction
from bcbio.variation import annotation
from bcbio.log import logger2 as logger


def _freebayes_options_from_config(aconfig):
    opts = []
    opts += ["--ploidy", str(aconfig.get("ploidy", 2))]
    regions = aconfig.get("variant_regions", None)
    if regions:
        opts += ["--targets", regions]
    background = aconfig.get("call_background", None)
    if background:
        opts += ["--variant-input", background]
    return opts


def run_freebayes(align_bam, ref_file, config, dbsnp=None, region=None,
                  out_file=None):
    """Detect small polymorphisms with FreeBayes.
    """
    if out_file is None:
        out_file = "%s-variants.vcf" % os.path.splitext(align_bam)[0]

    if not file_exists(out_file):
        logger.info("Genotyping with FreeBayes: {region} {fname}".format(
            region=region, fname=os.path.basename(align_bam)))
        with file_transaction(out_file) as tx_out_file:
            cl = [config["program"].get("freebayes", "freebayes"),
                  "-b", align_bam, "-v", tx_out_file, "-f", ref_file,
                  "--left-align-indels"]
            cl += _freebayes_options_from_config(config["algorithm"])
            if region:
                cl.extend(["-r", region])

            subprocess.check_call(cl)

    return out_file


def postcall_annotate(in_file, ref_file, vrn_files, config):
    """Perform post-call annotation of FreeBayes calls in preparation for filtering.
    """
    out_file = annotation.annotate_nongatk_vcf(in_file, vrn_files.dbsnp,
                                               ref_file, config)
    return out_file


def _check_file_gatk_merge(vcf_file):
    """Remove problem lines generated by GATK merging from FreeBayes calls.

    Works around this issue until next GATK release:
    http://getsatisfaction.com/gsa/topics/
    variantcontext_creates_empty_allele_from_vcf_input_with_multiple_alleles
    """
    def _not_empty_allele(line):
        parts = line.split("\t")
        alt = parts[4]
        return not alt[0] == ","
    orig_file = "{0}.orig".format(vcf_file)
    if not file_exists(orig_file):
        shutil.move(vcf_file, orig_file)
        with open(orig_file) as in_handle:
            with open(vcf_file, "w") as out_handle:
                for line in in_handle:
                    if line.startswith("#") or _not_empty_allele(line):
                        out_handle.write(line)
    return vcf_file
