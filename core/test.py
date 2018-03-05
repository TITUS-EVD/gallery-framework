from ROOT import galleryfmwk, larutil

larutil.LArUtilManager.Reconfigure(galleryfmwk.geo.kSBND)

geom = larutil.Geometry.GetME()
detProp = larutil.DetectorProperties.GetME()

print "Det Width: " + str(2* geom.DetHalfWidth())
print "Det Height: " + str(2* geom.DetHalfHeight())
print "Det Length: " + str(geom.DetLength())

print geom.Nwires(0)
print geom.Nwires(1)
print geom.Nwires(2)

print ""

print "SamplingRate [ns]: " + str(detProp.SamplingRate())
print "NumberTimeSamples: " + str(detProp.NumberTimeSamples())
print "ReadOutWindowSize: " + str(detProp.ReadOutWindowSize())
print ""


geoHelper = larutil.GeometryHelper.GetME()
larp = larutil.LArProperties.GetME()
print "Drift Velocity [cm/us]: " + str(larp.DriftVelocity())
print "Time to CM: " + str(geoHelper.TimeToCm())
print "Drift Field [kV/cm]: " + str(larp.Efield())

print "N_ticks drift window: " + str(2*geom.DetHalfWidth() / geoHelper.TimeToCm() )
print "Total distance [cm]: " + str(geoHelper.TimeToCm() * detProp.ReadOutWindowSize() )

val = larp.DriftVelocity() * (1e-3 * detProp.SamplingRate() ) * 4 * 512
print "Value is: " + str(val)


# drift_velocity_in_cm/us x 
# sampling_period_in_us x 
# row_compression_factor x 
# number_rows_in_image