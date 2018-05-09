import csv

def get_acronyms():
    y_mapping = dict()
    acronym_ordered = list()
    with open('tybalt_features_with_clinical.tsv') as tsvfile:
        reader = csv.DictReader(tsvfile, dialect='excel-tab')
        for row in reader:
            #print(row)
            y_mapping[row['sample_id']] = row['acronym']
            acronym_ordered.append(row['sample_id'])

    return y_mapping, acronym_ordered



if __name__ == '__main__':
    get_acronyms()