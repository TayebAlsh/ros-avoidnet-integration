import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/risc/rov/src/image_processor_pkg/install/image_processor_pkg'
