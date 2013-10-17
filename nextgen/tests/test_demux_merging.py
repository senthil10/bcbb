import os
import unittest
import logging
import sys
import re
import shutil
from xml.etree import ElementTree as ET
from xml.etree.ElementTree import tostring
from os.path import join as pjoin
from bs4 import BeautifulSoup
from nose.plugins.attrib import attr

from bcbio.utils import merge_demux_results


class DemuxFilesMerging(unittest.TestCase):
    """Test all functions to merge Demultiplexing result files.
    """
    def setUp(self):
        #Set up logging
        FORMAT = '[%(asctime)s] %(message)s'
        logging.basicConfig(format=FORMAT, level = logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(FORMAT))
        self.logger = logging.getLogger('Logger')
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(handler)

        #Common vars
        self.fc_dir = pjoin(os.path.dirname(__file__), "data",
                                   "130801_test_demux_AFC12345AAXX")
        self.unaligned_expected = pjoin(self.fc_dir, '_Unaligned_expected')
        self.unaligned = pjoin(self.fc_dir, 'Unaligned')
        self.fc_id = os.path.basename(self.fc_dir).split('_')[-1][1:]
        self.basecall_dir = 'Basecall_Stats_{}'.format(self.fc_id)
        self.expected_flowcell_demux_summary = ET.parse(
            pjoin(self.unaligned_expected, self.basecall_dir,
                         'Flowcell_demux_summary.xml'))

    def tearDown(self):
        shutil.rmtree(self.unaligned)

    def _compare_flowcell_demux_summary(self):
        """Compare the elements in two Flowcell_demux_summary.xml files.
        """
        #The expected file is already sorted by lane, but we have to sort the
        #merged one in order to compare them
        fc1 = ET.parse(pjoin(self.unaligned, self.basecall_dir,
            'Flowcell_demux_summary.xml'))
        fc2 = ET.parse(pjoin(self.unaligned_expected, self.basecall_dir,
            'Flowcell_demux_summary.xml'))
        r_merged = fc1.getroot()
        r_expected = fc2.getroot()

        #Compare the content of the XML (trim spaces and newlines)
        return re.sub('\s+', '', tostring(r_expected)).strip() == \
                        re.sub('\s+', '', tostring(r_merged)).strip()

    def _compare_demultiplex_stats(self):
        """Compare the elements in two Demultiplex_Stats.htm files.
        """
        with open(pjoin(self.unaligned, self.basecall_dir,
            'Demultiplex_Stats.htm')) as f:
            ds_merged = BeautifulSoup(f.read())
        with open(pjoin(self.unaligned_expected, self.basecall_dir,
            'Demultiplex_Stats.htm')) as f:
            ds_expected = BeautifulSoup(f.read())

        #Compare the content of the htm
        return re.sub('\s+', '', ds_merged.renderContents()) == \
                        re.sub('\s+', '', ds_expected.renderContents())


    def _compare_undemultiplexed_stats(self):
        """Compare if two Undemultiplexed_stats.metrics are equal
        """
        with open(pjoin(self.unaligned, self.basecall_dir,
            'Undemultiplexed_stats.metrics')) as f:
            us_merged = f.readlines()
        with open(pjoin(self.unaligned_expected, self.basecall_dir,
            'Undemultiplexed_stats.metrics')) as f:
            us_expected = f.readlines()
        return us_merged == us_expected


    def test_merge_unaligned_folder(self):
        """Merging Unaligned folders and comparing the results with the expected
        """
        self.logger.info("Merging Unaligned folders")
        merge_demux_results(self.fc_dir)
        self.logger.info("Testing the merging of Flowcell_demux_summary.xml")
        self.assertTrue(self._compare_flowcell_demux_summary(),
            "The resulting file Flowcell_demux_summary.xml is not as expected.")
        self.logger.info("Testing the merging of Demultiplex_Stats.htm")
        self.assertTrue(self._compare_demultiplex_stats(),
            "The resulting file Demultiplex_Stats.htm is not as expected.")
        self.logger.info("Testing the merging of Undemultiplexed_stats.metrics")
        self.assertTrue(self._compare_undemultiplexed_stats(),
            "The resulting file Undemultiplexed_stats.metrics is not as expected")
