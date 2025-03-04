import numpy as np
from encoder import XML2Dict

def parse_xml2mat(xml_path):
    xml = XML2Dict()
    with open(xml_path, 'r', encoding='utf-8') as f:
        s = f.read()   
        xml_dict = xml.parse(s)

    mat_str = xml_dict['trakem2']['t2_layer_set']['t2_layer'][1]['@t2_patch']['transform'][7:-1]
    width = int(xml_dict['trakem2']['t2_layer_set']['t2_layer'][0]['@t2_patch']['width'])
    height = int(xml_dict['trakem2']['t2_layer_set']['t2_layer'][0]['@t2_patch']['height'])
    mat_value = np.array(mat_str.split(','), dtype=np.float32)
    mat = np.eye(3)
    mat[0, :2] = mat_value[:2]
    mat[1, :2] = mat_value[2:4]
    mat[:2, 2] = mat_value[4:]
    
    return mat, [height, width]