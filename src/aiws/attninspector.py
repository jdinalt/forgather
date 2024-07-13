from aiws.dotdict import DotDict

class AttnInspector:
    def num_heads(self):
        return None

    def head_dims(self):
        return DotDict({ "query": None, "key": None })

    def alibi_slopes(self):
        return None

    def dot_product_scale(self):
        return None

    def weights(self, head=None):
        return DotDict({ "query": None, "key": None, "value": None, "output": None })

    def biases(self, head=None):
        return DotDict({ "query": None, "key": None, "value": None, "output": None })

class LayerInspector:
    def attention(self):
        return None

    def feedforward(self):
        return None

class ModelInspector:
    def embedding(self):
        return None

    def positional_encoder(self):
        return None

    def layer(self, l):
        return None