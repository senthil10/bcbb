import os
import unittest
import logging
import sys
import re
from xml.etree import ElementTree as ET
from xml.etree.ElementTree import tostring
from os.path import join as pjoin

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
        #remove spaces and newline characters to compare just the content of the
        #XML
        return re.sub('\s+', '', tostring(r_expected)).strip() == \
                        re.sub('\s+', '', tostring(r_merged)).strip()

    def test_merge_unaligned_folder(self):
        """Merging Unaligned folders and comparing the results with the expected
        """
        self.logger.info("Merging Unaligned folders")
        merge_demux_results(self.fc_dir)
        self.logger.info("Checking the file Flowcell_demux_summary.xml")
        self.assertTrue(self._compare_flowcell_demux_summary())
