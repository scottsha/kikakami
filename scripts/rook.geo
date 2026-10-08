// Capless cylinder with rook-style crenellation notches cut into
// the top rim.
// Axis along z, from z=0 to z=55, radius 10 mm (same as cylinder.geo)

SetFactory("OpenCASCADE");

lc = 0.30901699437495;
r  = 10;
h  = 55;

// Notch parameters (variable)
width        = 1.5;   // half-width of each notch box, along y
depth        = 2 * width;   // half-depth of each notch box, along z (centered on z = h)
num_rotations = 7;  // number of notches evenly spaced about the z axis

// Capless cylinder as a single lateral surface: extrude a full
// circle curve along +z (no cap surfaces are created)
Circle(1) = {0, 0, 0, r, 0, 2*Pi};
Extrude {0, 0, h} { Curve{1}; }

// One notch box per rotation, positioned at
// [0, 2r] x [-width, width] x [h - depth, h + depth],
// then rotated about the z axis by k * 2*pi/num_rotations
For k In {0 : num_rotations-1}
  angle = k * 2*Pi / num_rotations;
  Box(100 + k) = {0, -width, h - depth, 2*r, 2*width, 2*depth};
  Rotate {{0, 0, 1}, {0, 0, 0}, angle} { Volume{100 + k}; }
EndFor

notch_boxes[] = {100 : 100 + num_rotations - 1};
lateral[] = BooleanDifference{ Surface{1}; Delete; }{ Volume{notch_boxes[]}; Delete; };

Physical Surface("lateral") = {lateral[]};

Mesh.CharacteristicLengthMin = lc;
Mesh.CharacteristicLengthMax = lc;

Mesh 2;