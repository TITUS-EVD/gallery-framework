#ifdef __CINT__
#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;

#pragma link C++ namespace titusio;
#pragma link C++ class titusio::HitColumns;
#pragma link C++ class titusio::WireColumns;
#pragma link C++ class titusio::RawDigitColumns;
#pragma link C++ function titusio::readHits;
#pragma link C++ function titusio::readWires;
#pragma link C++ function titusio::readRawDigits;
#pragma link C++ function titusio::readChannelROIs;
#pragma link C++ function titusio::hasChannelROISupport;
#pragma link C++ class titusio::TrackColumns;
#pragma link C++ class titusio::PointColumns;
#pragma link C++ class titusio::ShowerColumns;
#pragma link C++ class titusio::PFParticleColumns;
#pragma link C++ class titusio::T0Columns;
#pragma link C++ class titusio::AssociationColumns;
#pragma link C++ function titusio::readTracks;
#pragma link C++ function titusio::readVertices;
#pragma link C++ function titusio::readSpacePoints;
#pragma link C++ function titusio::readShowers;
#pragma link C++ function titusio::readPFParticles;
#pragma link C++ function titusio::readT0s;
#pragma link C++ function titusio::readAssociation;
#endif
