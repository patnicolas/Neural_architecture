__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2020, 2022  All rights reserved."

from kafka import KafkaProducer


class KProducer(object):
    def __init__(self):
        self.producer = KafkaProducer()


    def produce(self, topic: str, key: str, value: object):
        self.producer.produce(topic, key, value)
