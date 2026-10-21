// Units: mm. No printed threads. Edit source JSON and rebuild to keep sim aligned.
// openscad -o base.stl -D 'part="base"' fixture.scad
part = "assembly"; // base, rotor_round, rotor_hex, assembly
$fn = 96;
r = 25.0; h = 20.0; bore = 4.25;
flange_r = 45.0; flange_h = 8.0;
post_r = 10.0; post_h = 35.0;
head_af = 13.4; head_h = 5.6;
gap = 1.0;
module base() { difference() {
    union() { cylinder(r=flange_r,h=flange_h); translate([0,0,flange_h]) cylinder(r=post_r,h=post_h); }
    translate([0,0,-0.1]) cylinder(r=bore,h=flange_h+post_h+0.2);
    translate([0,0,-0.1]) rotate([0,0,30]) cylinder(r=head_af/sqrt(3),h=head_h+0.1,$fn=6);
} }
module rotor(hex=false) { difference() {
    if (hex) rotate([0,0,30]) cylinder(r=r,h=h,$fn=6); else cylinder(r=r,h=h);
    translate([0,0,-0.1]) cylinder(r=bore,h=h+0.2);
} }
if (part=="base") base();
else if (part=="rotor_round") rotor();
else if (part=="rotor_hex") rotor(true);
else { base(); translate([0,0,flange_h+post_h+gap]) rotor(); }
