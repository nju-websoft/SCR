import torch
from configparser import ConfigParser
from transformers import BertTokenizer, AutoTokenizer

class Config(ConfigParser):
    def __init__(self, file):
        self.configParser = ConfigParser()
        self.configParser.read(file)
        self.load_value()

    def load_value(self):
        for section in self.configParser.sections():
            for key, value in self.configParser.items(section):
                val = None
                for attr in ['getint', 'getfloat', 'getboolean']:
                    try:
                        val = getattr(self.configParser[section], attr)(key)
                        break
                    except:
                        val = value
                assert(val!=None)
                setattr(self, key, val)
                print(key, val)
