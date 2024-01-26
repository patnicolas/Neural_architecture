__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2020, 2022  All rights reserved."

from abc import abstractmethod
from kafka import KafkaConsumer
from kafka.errors import KafkaError
import sys


class KConsumer(object):
    MIN_COMMIT_COUNT = 128

    def __init__(self, kafka_consumer_prop: dict, topic: str, polling_interval_ms: int):
        assert(20 < polling_interval_ms < 8096, f'Polling interval {polling_interval_ms} is out of range ]20, 8096[')
        self.consumer = KafkaConsumer(topic, group_id='group-1', bootstrap_servers=['localhost:9092'])
        self.consumer.subscribe([topic])
        self.polling_interval_ms = polling_interval_ms
        self.running = True

    @classmethod
    def build(cls, property_file: str, is_property_file_valid: bool) -> object:
        """
        Alternative constructor using an existing property file
        :param property_file: Name of the property file. This should be replaced by S3 or RDS connectivity
        :param is_property_file_valid Flag to specify that the property file has to be validated
        :return: Instance of KafkaConsumer
        """
        # load method to be implemented
        kafka_consumer_prop, topic, polling_interval_ms = load(property_file)
        return cls(kafka_consumer_prop, topic, polling_interval_ms)

    @classmethod
    def auto_build(cls, s3_bucket: str, s3_property_file: str, is_property_file_valid: bool) -> object:
        """
        Alternative constructor using an existing property file
        :param s3_bucket: Name of S3 bucket
        :param s3_property_file: Name of the property file.
        :param is_property_file_valid Flag to specify that the property file has to be validated
        :return: Instance of KafkaConsumer
        """
        # load method to be implemented
        kafka_consumer_prop, topic, polling_interval_ms = load(s3_bucket, s3_property_file)
        return cls(kafka_consumer_prop, topic, polling_interval_ms)

    def consume(self):
        msg_count = 0
        try:
            # It worth looking
            while self.running:
                msg = self.consumer.poll(self.polling_interval_ms)
                if msg:
                    if msg.values():
                        if msg.values() == KafkaError:
                            # Should be replaced by a message to be produced in error queue.
                            sys.stderr.write\
                                ('%% %s [%d] end of offset %d\n' % (msg.topic(), msg.partition(), msg.offset()))
                        # It should be caught and produce a message to Kafka error queue.
                        elif msg.error():
                            raise Exception(msg.error())
                    else:
                        self.process(msg)
                        msg_count += 1
                        if msg_count % self.MIN_COMMIT_COUNT == 0:
                            self.consumer.commit(asychronous=True)

        finally:
            self.consumer.close()
            self.running = False

    @abstractmethod
    def process(self, msg) -> bool:
        """
        Should be overwritten in sub classes
        :param msg: Group of messages consumed by the previous polling
        :return:  True if no HTTP 500 internal error, False otherwise
        """
        pass

    def shutdown(self):
        self.running = False
