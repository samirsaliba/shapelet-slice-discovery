class ShapeletIndividual(list):
    def __init__(self, *args):
        super().__init__(*args)
        self.in_top_k = False
        self.valid = False
        self.subgroup = None
        self.info = None
