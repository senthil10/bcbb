import unittest
import shutil
import tempfile
import os
import sys
import csv
import xml.etree.ElementTree as ET
from itertools import izip
from bcbio import utils
import illumina_finished_msg as ifm

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
        self.assertRaises(ValueError, ifm.initial_processing, *args, **self.kwargs)

    def test_call_as_in_process_first_read(self):
        args = ["", None, "", ""]  # [dname, config, local_config, unaligned_dir]
        self.assertRaises(OSError, ifm._post_process_run, *args, **self.kwargs)


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
                s4 = sample.format(index='ACGTAG-ACGTAG').split(',')
                rows = [s1, s2, s3, s4]
            elif index_info == 'no_index':
                s1 = sample.format(index='').split(',')
                s2 = sample.format(index='NoIndex').split(',')
                rows = [s1,s2]
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
        self.assertFalse(ifm._is_finished_first_base_report(self.rootdir))
        utils.touch_file(os.path.join(self.rootdir,"First_Base_Report.htm"))
        self.assertTrue(ifm._is_finished_first_base_report(self.rootdir))

    def test__is_started_initial_processing(self):
        """Initial processing started"""
        self.assertFalse(ifm._is_started_initial_processing(self.rootdir))
        utils.touch_indicator_file(os.path.join(self.rootdir,"initial_processing_started.txt"))
        self.assertTrue(ifm._is_started_initial_processing(self.rootdir))

    def test__is_started_first_read_processing(self):
        """First read processing started
        """
        self.assertFalse(ifm._is_started_first_read_processing(self.rootdir))
        utils.touch_indicator_file(os.path.join(self.rootdir,"first_read_processing_started.txt"))
        self.assertTrue(ifm._is_started_first_read_processing(self.rootdir))

    def test__is_started_second_read_processing(self):
        """Second read processing started
        """
        self.assertFalse(ifm._is_started_second_read_processing(self.rootdir))
        utils.touch_indicator_file(os.path.join(self.rootdir,"second_read_processing_started.txt"))
        self.assertTrue(ifm._is_started_second_read_processing(self.rootdir))

    def test__is_initial_processing(self):
        """Initial processing in progress"""
        self.assertFalse(ifm._is_initial_processing(self.rootdir),
                         "No indicator files should not indicate processing in progress")
        utils.touch_indicator_file(os.path.join(self.rootdir,"initial_processing_started.txt"))
        self.assertTrue(ifm._is_initial_processing(self.rootdir),
                        "Started indicator file should indicate processing in progress")
        utils.touch_indicator_file(os.path.join(self.rootdir,"initial_processing_completed.txt"))
        self.assertFalse(ifm._is_initial_processing(self.rootdir),
                        "Completed indicator file should not indicate processing in progress")

    def test__is_processing_first_read(self):
        """First read processing in progress
        """
        self.assertFalse(ifm._is_processing_first_read(self.rootdir),
                         "No indicator files should not indicate processing in progress")
        utils.touch_indicator_file(os.path.join(self.rootdir,"first_read_processing_started.txt"))
        self.assertTrue(ifm._is_processing_first_read(self.rootdir),
                        "Started indicator file should indicate processing in progress")
        utils.touch_indicator_file(os.path.join(self.rootdir,"first_read_processing_completed.txt"))
        self.assertFalse(ifm._is_processing_first_read(self.rootdir),
                        "Completed indicator file should not indicate processing in progress")

    def test__do_initial_processing(self):
        """Initial processing logic
        """
        self.assertFalse(ifm._do_initial_processing(self.rootdir),
                         "Initial processing should not be run with missing indicator flags")
        utils.touch_file(os.path.join(self.rootdir,"First_Base_Report.htm"))
        self.assertTrue(ifm._do_initial_processing(self.rootdir),
                         "Initial processing should be run after first base report creation")
        utils.touch_indicator_file(os.path.join(self.rootdir,"initial_processing_started.txt"))
        self.assertFalse(ifm._do_initial_processing(self.rootdir),
                         "Initial processing should not be run when processing has been started")
        os.unlink(os.path.join(self.rootdir,"First_Base_Report.htm"))
        self.assertFalse(ifm._do_initial_processing(self.rootdir),
                         "Initial processing should not be run when processing has been started " \
                         "and missing first base report")

    def test__do_first_read_processing(self):
        """First read processing logic
        """
        runinfo = os.path.join(self.rootdir, "RunInfo.xml")
        self._runinfo(runinfo)
        self.assertFalse(ifm._do_first_read_processing(self.rootdir),
                         "Processing should not be run before first read is finished")
        utils.touch_file(os.path.join(self.rootdir,
                                      "Basecalling_Netcopy_complete_Read1.txt"))
        self.assertFalse(ifm._do_first_read_processing(self.rootdir),
                         "Processing should not be run before last index read is finished")
        utils.touch_file(os.path.join(self.rootdir,
                                      "Basecalling_Netcopy_complete_Read2.txt"))
        utils.touch_indicator_file(os.path.join(self.rootdir,
                                                "initial_processing_started.txt"))
        self.assertFalse(ifm._do_first_read_processing(self.rootdir),
                         "Processing should not be run when previous processing step is in progress")
        utils.touch_indicator_file(os.path.join(self.rootdir,
                                                "initial_processing_completed.txt"))
        self.assertTrue(ifm._do_first_read_processing(self.rootdir),
                        "Processing should be run when last index read is finished")
        utils.touch_indicator_file(os.path.join(self.rootdir,
                                                "first_read_processing_started.txt"))
        self.assertFalse(ifm._do_first_read_processing(self.rootdir),
                         "Processing should not be run when processing has started")

    def test__do_second_read_processing(self):
        """Second read processing logic
        """
        runinfo = os.path.join(self.rootdir, "RunInfo.xml")
        self._runinfo(runinfo)
        utils.touch_file(os.path.join(self.rootdir,
                                      "Basecalling_Netcopy_complete_READ2.txt"))
        self.assertTrue(ifm._do_second_read_processing(self.rootdir),
                        "Processing should be run when last read GAII checkpoint exists")
        os.unlink(os.path.join(self.rootdir,
                               "Basecalling_Netcopy_complete_READ2.txt"))
        self.assertFalse(ifm._do_second_read_processing(self.rootdir),
                         "Processing should not be run before any reads are finished")
        utils.touch_file(os.path.join(self.rootdir,
                                      "Basecalling_Netcopy_complete_Read2.txt"))
        self.assertFalse(ifm._do_second_read_processing(self.rootdir),
                         "Processing should not be run before last read is finished")
        utils.touch_file(os.path.join(self.rootdir,
                                      "Basecalling_Netcopy_complete_Read3.txt"))
        self.assertTrue(ifm._do_second_read_processing(self.rootdir),
                        "Processing should be run when last read is finished")
        utils.touch_indicator_file(os.path.join(self.rootdir,
                                                "second_read_processing_started.txt"))
        self.assertFalse(ifm._do_second_read_processing(self.rootdir),
                         "Processing should not be run when processing has started")

    def test__expected_reads(self):
        """Get expected number of reads
        """
        self.assertEqual(ifm._expected_reads(self.rootdir),0,
                         "Non-existant RunInfo.xml should return 0 expected reads")

        runinfo = os.path.join(self.rootdir,"RunInfo.xml")
        self._runinfo(runinfo)
        self.assertEqual(ifm._expected_reads(self.rootdir),3,
                         "Default RunInfo.xml should return 3 expected reads")

        self._runinfo(runinfo, "Y101,I6,I6,Y101")

        self.assertEqual(ifm._expected_reads(self.rootdir),4,
                         "Dual-index RunInfo.xml should return 4 expected reads")

    def test__last_index_read(self):
        """Get number of last index read
        """
        self.assertEqual(ifm._last_index_read(self.rootdir),0,
                         "Non-existant RunInfo.xml should return 0 as last index read")

        runinfo = os.path.join(self.rootdir,"RunInfo.xml")
        self._runinfo(runinfo)
        self.assertEqual(ifm._last_index_read(self.rootdir),2,
                         "Default RunInfo.xml should return 2 as last index read")

        self._runinfo(runinfo, "Y101,I6,I6,Y101")
        self.assertEqual(ifm._last_index_read(self.rootdir),3,
                         "Dual-index RunInfo.xml should return 3 as last expected read")

        self._runinfo(runinfo, "Y101,Y101,Y101")
        self.assertEqual(ifm._last_index_read(self.rootdir),0,
                         "Non-index RunInfo.xml should return 0 as last expected read")

    def test__is_finished_basecalling_read(self):
        """Detect finished read basecalling
        """

        # Create a custom RunInfo.xml in the current directory
        runinfo = os.path.join(self.rootdir,"RunInfo.xml")
        self._runinfo(runinfo, "Y101,Y101")

        with self.assertRaises(ValueError):
            ifm._is_finished_basecalling_read(self.rootdir,0)

        with self.assertRaises(ValueError):
            ifm._is_finished_basecalling_read(self.rootdir,3)

        for read in (1,2):
            self.assertFalse(ifm._is_finished_basecalling_read(self.rootdir,read),
                             "Should not return true with missing indicator file")
            utils.touch_file(os.path.join(self.rootdir,
                                          "Basecalling_Netcopy_complete_Read{:d}.txt".format(read)))
            self.assertTrue(ifm._is_finished_basecalling_read(self.rootdir,read),
                            "Should return true with existing indicator file")

    def test__get_bases_mask(self):
        """Get bases mask
        """
        runinfo = os.path.join(self.rootdir, "RunInfo.xml")
        self._runinfo(runinfo)
        flowcell_id = ifm._get_flowcell_id(self.rootdir)
        samplesinfo = os.path.join(self.rootdir, str(flowcell_id) + '.csv')
        #Test simple index
        self._samplesinfo(samplesinfo)
        self.assertEqual(ifm._get_bases_mask(self.rootdir)[6]['base_mask'], ['Y101', 'I6N1', 'Y101'])
        #Test mixed indexes
        self._runinfo(runinfo, bases_mask='Y101,I8,I8,Y101')
        self._samplesinfo(samplesinfo, index_info='mixed_index')
        masks = ifm._get_bases_mask(self.rootdir)
        self.assertEqual(masks[6]['base_mask'], ['Y101', 'I6N2', 'N8', 'Y101'])
        self.assertEqual(masks[8]['base_mask'], ['Y101', 'I8', 'N8', 'Y101'])
        self.assertEqual(masks[12]['base_mask'], ['Y101', 'I6N2', 'I6N2', 'Y101'])
        self.assertEqual(masks[16]['base_mask'], ['Y101', 'I8', 'I8', 'Y101'])
        #Test no index
        self._runinfo(runinfo, bases_mask='Y101,I8,I8,Y101')
        self._samplesinfo(samplesinfo, index_info='no_index')
        masks = ifm._get_bases_mask(self.rootdir)
        self.assertEqual(masks[0]['base_mask'], ['Y101', 'Y8', 'Y8','Y101'])
        self.assertEqual(len(masks), 1, "Only one index length is expected")



    def test__get_read_configuration(self):
        """Get read configuration
        """

        self.assertListEqual(ifm._get_read_configuration(self.rootdir), [],
                             "Expected empty list for non-existing RunInfo.xml")

        runinfo = os.path.join(self.rootdir,"RunInfo.xml")
        self._runinfo(runinfo)
        obs_reads = ifm._get_read_configuration(self.rootdir)
        self.assertListEqual([r.get("Number",0) for r in obs_reads],["1","2","3"],
                             "Expected 3 reads for 2x100 PE")

    def test__get_directories(self):
        """Get run output directories
        """
        config = {"dump_directories": [self.rootdir]}
        obs_dirs = [d for d in ifm._get_directories(config)]
        self.assertListEqual([],obs_dirs,
                              "Expected empty list for getting non-existing run directories")
        utils.touch_file(os.path.join(self.rootdir, "111111_SN111_1111_A11111111"))
        obs_dirs = [d for d in ifm._get_directories(config)]
        self.assertListEqual([],obs_dirs,
                              "Should not pick up files, only directories")
        exp_dirs = [os.path.join(self.rootdir, "222222_SN222_2222_A2222222"),
                    os.path.join(self.rootdir, "333333_D0023_3333_B33333XX")]
        os.mkdir(exp_dirs[-1])
        os.mkdir(exp_dirs[-2])
        obs_dirs = [d for d in ifm._get_directories(config)]
        self.assertListEqual(sorted(exp_dirs),sorted(obs_dirs),
                              "Should pick up matching directory - hiseq-style")
        exp_dirs.append(os.path.join(self.rootdir, "333333_M33333_3333_A000000000-A3333"))
        os.mkdir(exp_dirs[-1])
        obs_dirs = [d for d in ifm._get_directories(config)]
        self.assertListEqual(sorted(exp_dirs),sorted(obs_dirs),
                              "Should pick up matching directory - miseq-style")
