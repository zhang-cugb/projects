class UnitRegistry:
    '''Registry that containts the conversions\
    from all units to SI units'''

    def __init__(self):
        # International System Units
        self.meter = 1
        self.second = 1
        self.kilogram = 1
        self.pascal = 1
        # NOT SI
        self.poise = 0.1  # Pa x s
        self.darcy = 9.869233E-13  # m2
        # Field Units
        self.feet = 0.3048  # m
        self.inches = 0.0254  # m
        self.pound = 0.453592  # kg
        self.psi = 6894.76  # Pa
        self.barrel = 0.1589873  # m3
        self.gallon = 0.00378541  # m3
        self.minute = 60  # s
        self.hour = 3600  # s
        self.day = 86400  # s
        # Prefixes
        self.milli = 1E-3
        self.centi = 1E-2
        self.deci = 1E-1
        self.deka = 1E1
        self.hecto = 1E2
        self.kilo = 1E3


class Constants:
    ''' Registry with the physical constants in SI units.
    Attributes:
    g  : gravity constant
    '''
    
    def __init__(self):
        self.g = 9.81  # m/s^2
