"""
This file gets the dataset and removes unnecessary new lines.
"""


def boston_data_customiser(data):
    """ This function removes new lines from even numbered rows, and saves in new file: boston_new.csv,

    Args:
        data ([data]): [this is boston data's first view]
    """
    with open(data) as d_line:
        text = [line for line in d_line.readlines()]
        start_row = 0
    new_rows = []
    for i, l in enumerate(text[start_row:]):
        if not i % 2:
            newl = l.strip('\n')+text[start_row+i+1]
            new_rows.append(newl)

    new_data = ''.join(new_rows)

    with open('/Users/armenhakobyan/PycharmProjects/BostonProject/dataset/boston_new.csv', 'w') as f:
        f.write(new_data)


boston_data_customiser('/Users/armenhakobyan/PycharmProjects/BostonProject/dataset/Boston.txt')
