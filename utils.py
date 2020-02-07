import re

def get_class_name_from_string(image_path):
    """
    parse string to get class name
    """
    spe_char_1 = '/'
    spe_char_2 = '\\'
    spe_char_3 = '_'
    image_path = str(image_path)
    char_1_pos= [pos for pos, char in enumerate(image_path) if char == spe_char_1]
    char_2_pos= [pos for pos, char in enumerate(image_path) if char == spe_char_2]
    last_spe_char_pos = char_1_pos[-1] if char_1_pos[-1] > char_2_pos[-1] else char_2_pos[-1]
    image_name = image_path[last_spe_char_pos + 1:len(image_path)]
    char_3_pos= [pos for pos, char in enumerate(image_name) if char == spe_char_3]
    class_name = image_name[0:char_3_pos[0]]
    return class_name
     