"""Work with Broad's Java libraries from Python.

  Picard -- BAM manipulation and analysis library.
  GATK -- Next-generation sequence processing.
"""
import os
import subprocess

from bcbio.broad import picardrun


class BroadRunner:
    """Simplify running Broad commandline tools.
    """
    def __init__(self, picard_ref, gatk_dir="", max_memory=None,
                 config=None):
        self._memory_args = []
        if not max_memory:
            max_memory = "6g"
        self._memory_args.append("-Xmx%s" % max_memory)
        self._picard_ref = picard_ref
        self._gatk_dir = gatk_dir or picard_ref
        if config is None:
            config = {}
        self._config = config

    def run_fn(self, name, *args, **kwds):
        """Run pre-built functionality that used Broad tools by name.

        See the gatkrun, picardrun module for available functions.
        """
        fn = None
        to_check = [picardrun]
        for ns in to_check:
            try:
                fn = getattr(ns, name)
                break
            except AttributeError:
                pass

        assert fn is not None, \
        "Could not find function %s in %s" % (name, to_check)

        return fn(self, *args, **kwds)

    def run(self, command, options):
        """Run a Picard command with the provided option pairs.
        """
        options = ["%s=%s" % (x, y) for x, y in options]
        options.append("VALIDATION_STRINGENCY=SILENT")
        cl = self._get_picard_cmd(command) + options
        subprocess.check_call(cl)

    def run_gatk(self, params, tmp_dir=None):
        support_parallel = ["UnifiedGenotyper", "CountCovariates", "VariantEval",
                            "VariantRecalibrator"]
        gatk_jar = self._get_jar("GenomeAnalysisTK")
        local_args = []
        cores = self._config.get("resources", {}).get("gatk", {}).get("cores", None)
        if cores:
            do_parallel = False
            for check in support_parallel:
                if check in params:
                    do_parallel = True

            if do_parallel:
                params.extend(["-nt", str(cores)])

        if tmp_dir:
            local_args.append("-Djava.io.tmpdir=%s" % tmp_dir)

        cl = ["java"] + self._memory_args + local_args + \
                ["-jar", gatk_jar] + [str(x) for x in params]
        subprocess.check_call(cl)

    def get_picard_version(self, command):
        cl = self._get_picard_cmd(command) + ["--version"]
        p = subprocess.Popen(cl, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        version = float(p.stdout.read().split("(")[0])
        p.wait()
        return version

    def _get_picard_cmd(self, command):
        """Retrieve the base Picard command, handling both shell scripts and directory of jars.
        """
        if os.path.isdir(self._picard_ref):
            dist_file = self._get_jar(command)
            return ["java"] + self._memory_args + ["-jar", dist_file]
        else:
            # XXX Cannot currently set JVM opts with picard-tools script
            return [self._picard_ref, command]

    def _get_jar(self, command):
        """Retrieve the jar for running the specified command.
        """
        dirs = []
        for bdir in [self._picard_ref, self._gatk_dir]:
            dirs.extend([bdir,
                         os.path.join(bdir, os.pardir, "gatk"),
                         os.path.join(bdir, "dist"),
                         os.path.join(bdir, "GATK"),
                         os.path.join(bdir, "GATK", "dist"),
                         os.path.join(bdir, "Picard-private", "dist")])

        for dir_check in dirs:
            check_file = os.path.join(dir_check, "%s.jar" % command)
            if os.path.exists(check_file):
                return check_file

        raise ValueError("Could not find jar %s in %s:%s" % (command, self._picard_ref, self._gatk_dir))


def runner_from_config(config):
    return BroadRunner(config["program"].get("picard", ""),
                       config["program"].get("gatk", ""),
                       max_memory=config["algorithm"].get("java_memory", ""),
                       config=config)
