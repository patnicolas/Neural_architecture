
from confluent_kafka import Producer


class KafkaProducer(object):
    def __init__(self, kafka_prop: dict):
        self.producer = Producer(kafka_prop)

    @classmethod
    def build(cls, kafka_prop_file):

        return cls()

    def produce(self, topic: str, key: str, value: object):
        self.producer.produce(topic, key, value)
