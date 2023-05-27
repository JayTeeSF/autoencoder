#!/usr/bin/env python

###import tensorflow as tf
###print(tf.sysconfig.get_build_info()["build_info"]["compiler_version"])
#
##import tensorflow as tf
##
##default_optimizer = tf.keras.optimizers.get({
##    "class_name": tf.keras.optimizers.get.__name__,
##    "config": {"name": "optimizer"}
##})
##
##print(default_optimizer.__class__.__name__)
#
#import tensorflow as tf
#
#default_optimizer = tf.keras.optimizers.get(tf.keras.optimizers.Adam)
#
#print(default_optimizer.__class__.__name__)

#import tensorflow as tf

#default_optimizer = tf.keras.optimizers.Adam

#print(default_optimizer.__name__)
import tensorflow as tf

# print(tf.keras.optimizers.experimental.Adam._DEFAULT_OPTIMIZER)
#print(tf.keras.optimizers.experimental.adam.__defaults__)
print(tf.keras.optimizers.experimental.Adam.__defaults__)

