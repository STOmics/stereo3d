import numpy as np
from encoder import XML2Dict
import tifffile as tif
import cv2 as cv


def parse_mls_data(data_str):
    parts = data_str.split()
    params = {
        'transform_type': parts[0],
        'dimension': int(parts[1]),
        'alpha': float(parts[2]),
        'src_points': [],
        'dst_points': []
    }
    
    # record points pairs
    n_points = (len(parts) - 3) // 5
    index = 3
    
    for i in range(n_points):
        src_x = float(parts[index])
        src_y = float(parts[index + 1])
        dst_x = float(parts[index + 2])
        dst_y = float(parts[index + 3])
        
        params['src_points'].append([src_x, src_y])
        params['dst_points'].append([dst_x, dst_y])
        index += 5
    
    return params

def parse_xml2mat(xml_path):
    xml = XML2Dict()
    with open(xml_path, 'r', encoding='utf-8') as f:
        s = f.read()   
        xml_dict = xml.parse(s)
    
    width = float(xml_dict['trakem2']['t2_layer_set']['t2_layer'][1]['@t2_patch']['width'])
    print(width)
    height = float(xml_dict['trakem2']['t2_layer_set']['t2_layer'][1]['@t2_patch']['height'])
    print(height)
    width, height = int(width), int(height)
    transform_dict = {'base_transform': None, 'transform_sequence': []} # dict for recording data for transform
    patch = xml_dict['trakem2']['t2_layer_set']['t2_layer'][1]['@t2_patch']
    if 'transform' in patch:
        transform_dict['base_transform'] = patch['transform'][7:-1] # basic transform matrix
    patch = xml_dict['trakem2']['t2_layer_set']['t2_layer'][1]['t2_patch']
    
    if patch:
        def extract_from_element(element):
            if isinstance(element, dict):
                # transform elements
                if 'class' in element and 'data' in element:
                    class_name = element['class']
                    transform_dict['transform_sequence'].append({'class':class_name, 'data': element['data']})
                for value in element.values():
                    extract_from_element(value)
                    
            elif isinstance(element, list):
                for item in element:
                    extract_from_element(item)
        
        if 'ict_transform_list' in patch:
            extract_from_element(patch['ict_transform_list'])
    mat_sequence = []
    if transform_dict['base_transform'] != None:
        mat_value = np.array(transform_dict['base_transform'].split(','), dtype=np.float32)
        mat = np.eye(3)
        mat[0, :2] = mat_value[:2]
        mat[1, :2] = mat_value[2:4]
        mat[:2, 2] = mat_value[4:]
        # transform to OpenCV template
        mat[0, 1] = -mat[0, 1]
        mat[1, 0] = -mat[1, 0]
        mat_sequence.append(mat)
    if transform_dict['transform_sequence']:
        for transform in transform_dict['transform_sequence']:
            if transform['class'] == 'mpicbg.trakem2.transform.AffineModel2D':
                mat_value = np.array(transform['data'].split(' '), dtype=np.float32)
                mat = np.eye(3)
                mat[0, :2] = mat_value[:2]
                mat[1, :2] = mat_value[2:4]
                mat[:2, 2] = mat_value[4:]

                '''mat[0, 1] = -mat[0, 1]
                mat[1, 0] = -mat[1, 0]'''
                mat_sequence.append(mat)
            if transform['class'] == 'mpicbg.trakem2.transform.MovingLeastSquaresTransform2':
                mat = np.eye(3)
                mls_params = parse_mls_data(transform['data'])

                src_pts = np.float32(mls_params['src_points'])
                dst_pts = np.float32(mls_params['dst_points'])
                if len(src_pts) == 3:
                    M = cv.getAffineTransform(src_pts, dst_pts)
                elif len(src_pts) == 4:
                    M = cv.getPerspectiveTransform(src_pts, dst_pts)
                elif len(src_pts) > 4:
                    M, _ = cv.estimateAffinePartial2D(src_pts, dst_pts)
                if M.shape[0] == 3:
                    mat_sequence.append(M)
                else:
                    mat[:2, :] = M
                    '''mat[0, 1] = -mat[0, 1]
                    mat[1, 0] = -mat[1, 0]'''
                    mat_sequence.append(mat)

    return mat_sequence, [height, width]      

    
    
#-----------
    mat_str = xml_dict['trakem2']['t2_layer_set']['t2_layer'][1]['@t2_patch']['transform'][7:-1]
    width = float(xml_dict['trakem2']['t2_layer_set']['t2_layer'][0]['@t2_patch']['width'])
    print(width)
    height = float(xml_dict['trakem2']['t2_layer_set']['t2_layer'][0]['@t2_patch']['height'])
    print(height)
    width, height = int(width), int(height)
    mat_value = np.array(mat_str.split(','), dtype=np.float32)
    mat = np.eye(3)
    mat[0, :2] = mat_value[:2]
    mat[1, :2] = mat_value[2:4]
    mat[:2, 2] = mat_value[4:]

    mat[0, 1] = -mat[0, 1]
    mat[1, 0] = -mat[1, 0]
    
    return mat, [height, width]

def image_transform(input_img, mat, width, height):
    print(sum(sum(input_img)))
    affine_matrix = mat[:2, :]
    transformed = cv.warpAffine(input_img, affine_matrix, (width, height))
    print(sum(sum(transformed)))
    return transformed


if __name__ == '__main__':
    mat, shape = parse_xml2mat(r"e:\03.users\wangaoli\data\raw_data\Drosophila_test\test_result\02.register\02.manual\A02183A2.xml")
    print(mat)
