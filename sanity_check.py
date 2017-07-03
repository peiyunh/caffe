import sys
sys.path.append('python')

import caffe

proto_file = 'pspnet/pspnet101_cityscapes_713.prototxt'
weight_file = 'pspnet/pspnet101_cityscapes.caffemodel'

net = caffe.Net(proto_file, weight_file, caffe.TEST)


