import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET
import argparse


def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (
                root.find('filename').text,
                int(root.find('size')[0].text),
                int(root.find('size')[1].text),
                member[0].text,
                int(member[4][0].text),
                int(member[4][1].text),
                int(member[4][2].text),
                int(member[4][3].text)
            )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df


def generate_csv_file(path_to_images, path_to_output_csv_file):
    xml_df = xml_to_csv(path_to_images)
    xml_df.to_csv(path_to_output_csv_file, index=None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Parse an .xml file into .csv")

    parser.add_argument(
		"--path_to_images",
		type=str,
		help="folder that contains images",
	    default="maskDataset/dummy"
	)

    parser.add_argument(
		"--path_to_csv_file",
		type=str,
		help="full path to file where you want to save your data",
	    default="maskDataset/dummy/annotations.csv"
	)

    args = parser.parse_args()

    generate_csv_file(args.path_to_images, args.path_to_csv_file)
