import ms5837

class DepthReader:
    def __init__(self):
        self.sensor = ms5837.MS5837_02BA()
        
        # We must initialize the sensor before reading it
        if not self.sensor.init():
            print("Sensor could not be initialized")
            exit(1)
        
        # We have to read values from sensor to update pressure and temperature
        if not self.sensor.read():
            print("Sensor read failed!")
            exit(1)
        
        self.sensor.setFluidDensity(ms5837.DENSITY_SALTWATER)
    
    def get_depth(self):
        return self.sensor.depth()
    
    def get_temperature(self):
        return self.sensor.temperature(ms5837.UNITS_Centigrade)
    
    def get_depth_and_temp(self):
        return (self.get_depth(), self.get_temperature())
    
