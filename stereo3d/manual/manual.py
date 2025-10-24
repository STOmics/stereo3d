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

def parse_xml2mat(xml_path, name):
    xml = XML2Dict()
    with open(xml_path, 'r', encoding='utf-8') as f:
        s = f.read()   
        xml_dict = xml.parse(s)
    file_sequence = 1
    shape_p = 0
    for i in range(len(xml_dict['trakem2']['t2_layer_set']['t2_layer'])):
        if name in xml_dict['trakem2']['t2_layer_set']['t2_layer'][i]['@t2_patch']['title']:
            file_sequence = i
            shape_p = i
            break
    '''if name in xml_dict['trakem2']['t2_layer_set']['t2_layer'][0]['@t2_patch']['title']:
        file_sequence = 0
        shape_p = 1'''
    
    width = float(xml_dict['trakem2']['t2_layer_set']['t2_layer'][shape_p]['@t2_patch']['width'])
    height = float(xml_dict['trakem2']['t2_layer_set']['t2_layer'][shape_p]['@t2_patch']['height'])
    width, height = int(width), int(height)
    transform_dict = {'base_transform': None, 'transform_sequence': []} # dict for recording data for transform
    patch = xml_dict['trakem2']['t2_layer_set']['t2_layer'][file_sequence]['@t2_patch']
    if 'transform' in patch:
        transform_dict['base_transform'] = patch['transform'][7:-1] # basic transform matrix
    patch = xml_dict['trakem2']['t2_layer_set']['t2_layer'][file_sequence]['t2_patch']
    
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
                #mat = np.eye(3)
                mls_params = parse_mls_data(transform['data'])

                '''src_pts = np.float32(mls_params['src_points'])
                dst_pts = np.float32(mls_params['dst_points'])
                p = np.array(src_pts).tolist()
                q = np.array(dst_pts).tolist()'''
                src_pts = mls_params['src_points']
                dst_pts = mls_params['dst_points']
                mat = [src_pts] +[dst_pts]
                '''if len(src_pts) == 3:
                    M = cv.getAffineTransform(src_pts, dst_pts)
                elif len(src_pts) == 4:
                    M = cv.getPerspectiveTransform(src_pts, dst_pts)
                elif len(src_pts) > 4:
                    M, _ = cv.estimateAffinePartial2D(src_pts, dst_pts)
                if M.shape[0] == 3:
                    mat_sequence.append(M)
                else:
                    mat[:2, :] = M
                    #mat[0, 1] = -mat[0, 1]
                    #mat[1, 0] = -mat[1, 0]'''
                mat_sequence.append(mat)

    return mat_sequence, [height, width]      


if __name__ == '__main__':
    mat, shape = parse_xml2mat(r"D:\stereo3d_data\demo\Drosophila_melanogaster_demo\test_manual\02.register\02.manual\A02183A5.xml","A02183A5")
    print(mat)
    print(shape)