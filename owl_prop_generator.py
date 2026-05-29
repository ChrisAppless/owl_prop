"""
Parametric Propeller Generator with Owl-Style Leading Edge Serrations
======================================================================
Run this script in Blender's Scripting workspace.
Press N in the 3D Viewport → "Propeller" tab.

Fixes v3:
  - Proper blade twist: chord and thickness axes both rotate by pitch_angle
  - Hub blend: blade root section is scaled to wrap the hub cylinder
  - Serrations are chordwise displacement (per airfoil point, not per span ring)
    so faces stay clean regardless of tooth count
  - Tooth width now correctly controls the flat valley width for all profiles
"""

import bpy, math
from bpy.props import (IntProperty, FloatProperty, EnumProperty,
                       BoolProperty, StringProperty, PointerProperty)
from bpy.types import Panel, Operator, PropertyGroup


# ─────────────────────────────────────────────────────────────────────────────
#  NACA 4-digit section  →  list of (x,y) normalised 0..1
# ─────────────────────────────────────────────────────────────────────────────

def naca4_points(code_int, n):
    """
    Returns upper + lower surface points as two lists of (x,y), LE→TE.
    Cosine-clustered x for smooth leading-edge curvature.
    """
    s   = str(code_int).zfill(4)
    m   = int(s[0]) / 100.0
    p   = int(s[1]) / 10.0
    t   = int(s[2:]) / 100.0

    xs = [0.5*(1 - math.cos(math.pi*i/(n-1))) for i in range(n)]

    upper, lower = [], []
    for x in xs:
        yt = 5*t*(0.2969*math.sqrt(x) - 0.1260*x - 0.3516*x**2
                  + 0.2843*x**3 - 0.1015*x**4)
        if m == 0 or p == 0:
            yc, dyc = 0.0, 0.0
        elif x < p:
            yc  = m/p**2*(2*p*x - x**2)
            dyc = 2*m/p**2*(p - x)
        else:
            yc  = m/(1-p)**2*(1 - 2*p + 2*p*x - x**2)
            dyc = 2*m/(1-p)**2*(p - x)

        th = math.atan(dyc)
        upper.append((x - yt*math.sin(th),  yc + yt*math.cos(th)))
        lower.append((x + yt*math.sin(th),  yc - yt*math.cos(th)))

    return upper, lower


# ─────────────────────────────────────────────────────────────────────────────
#  Serration profile  (t in [0,1] along span → amplitude in [0,1])
# ─────────────────────────────────────────────────────────────────────────────

def serr_sine(t, w):
    """
    Smooth sinusoid. w = fraction of period that is the FLAT VALLEY.
    Peak occupies (1-w) of the period; valley occupies w.
    """
    # Remap t so the valley is centred at t=0
    # Peak at t=0.5*(1-w), valley spanning t=1-w..1 (mod 1)
    peak_frac = 1.0 - w
    c = t % 1.0
    if c < peak_frac:
        # Map 0..peak_frac → 0..1 for a half-sine bump
        u = c / peak_frac
        return math.sin(math.pi * u)
    else:
        return 0.0   # flat valley

def serr_triangle(t, w):
    """
    Triangle tooth. w = valley flat fraction.
    """
    peak_frac = 1.0 - w
    c = t % 1.0
    if c < peak_frac:
        u = c / peak_frac          # 0..1
        return 1.0 - abs(2*u - 1)  # triangle: 0→1→0
    else:
        return 0.0

def serr_sawtooth(t, w):
    """Asymmetric sawtooth (steep front, gentle back)."""
    peak_frac = 1.0 - w
    c = t % 1.0
    if c < peak_frac:
        return c / peak_frac
    else:
        return 0.0

PROFILES = {'SINUSOIDAL': serr_sine, 'TRIANGLE': serr_triangle, 'SAWTOOTH': serr_sawtooth}


# ─────────────────────────────────────────────────────────────────────────────
#  Section builder
#  Places one airfoil cross-section ring at radius r with correct twist.
#  Returns list of (x,y,z) world verts.
#
#  Coordinate system:
#    Z  = propeller axis (thrust direction)
#    XY = disc plane  (propeller rotates around Z)
#    Each section sits at angle `az` in the XY plane (for multi-blade rotation)
#    and is twisted by pitch_angle around the radial axis.
# ─────────────────────────────────────────────────────────────────────────────

def section_world_verts(r, chord, pitch_angle, az,
                        upper, lower,
                        serr_amp, serr_fn, serr_width,
                        t_serr, n_teeth):
    """
    Compute the world-space vertices for one airfoil cross-section.

    t_serr is the span position remapped so 0 = start of serration zone,
    ensuring the first tooth always begins cleanly at zero amplitude.

    Returns list of (x,y,z).
    """
    loop = upper + lower[-2:0:-1]   # full closed profile, 2*n_pts-2 points
    LE_FRAC   = 0.25                # front 25 % of chord receives serrations
    span_phase = t_serr * n_teeth

    # Radial axis direction (blade points outward at angle az in XY plane)
    rad_x = math.cos(az)
    rad_y = math.sin(az)

    # Chord direction (tangential, rotated by pitch_angle around radial axis)
    # In the section's local frame:  chord → tangential, thickness → axial (Z)
    # After pitch rotation around the radial axis:
    #   chord_world  =  (-sin(az), cos(az), 0) * cos(pa)  +  (0,0,1) * (-sin(pa))
    #   thick_world  =  (-sin(az), cos(az), 0) * sin(pa)  +  (0,0,1) * ( cos(pa))
    tan_x = -math.sin(az);  tan_y = math.cos(az)   # tangential unit vector

    chord_wx = tan_x * math.cos(pitch_angle)
    chord_wy = tan_y * math.cos(pitch_angle)
    chord_wz =        -math.sin(pitch_angle)

    thick_wx = tan_x * math.sin(pitch_angle)
    thick_wy = tan_y * math.sin(pitch_angle)
    thick_wz =         math.cos(pitch_angle)

    verts = []
    for (xn, yn) in loop:
        # Local chord-space: origin at quarter-chord, x along chord, y = thickness
        xc = (xn - 0.25) * chord
        yc = yn * chord

        # Serration on leading-edge region
        if serr_amp > 0 and xn < LE_FRAC:
            blend_c = 1.0 - xn / LE_FRAC
            tooth   = serr_fn(span_phase, serr_width) * serr_amp
            xc     -= tooth * blend_c

        # World position = section centre + chord displacement + thickness displacement
        wx = rad_x * r  +  chord_wx * xc  +  thick_wx * yc
        wy = rad_y * r  +  chord_wy * xc  +  thick_wy * yc
        wz =               chord_wz * xc  +  thick_wz * yc

        verts.append((wx, wy, wz))

    return verts


def project_to_cylinder(verts, hub_r):
    """
    Project each vertex radially onto the hub cylinder surface (radius hub_r).
    The Z coordinate is preserved; the XY direction is kept but the radius
    is clamped to hub_r.  Returns a new list of (x,y,z).
    """
    result = []
    for (x, y, z) in verts:
        rxy = math.sqrt(x*x + y*y)
        if rxy < 1e-6:
            result.append((hub_r, 0.0, z))
        else:
            scale = hub_r / rxy
            result.append((x * scale, y * scale, z))
    return result


# ─────────────────────────────────────────────────────────────────────────────
#  Full propeller mesh builder
# ─────────────────────────────────────────────────────────────────────────────

def build_propeller(pg):
    diameter    = pg.diameter
    pitch       = pg.pitch
    hub_r       = pg.hub_radius
    tip_r       = diameter / 2.0
    n_sec       = pg.sections
    n_pts       = pg.airfoil_pts
    chord_root  = pg.chord_root
    chord_tip   = pg.chord_tip
    num_blades  = pg.num_blades

    serr_on  = pg.serr_enabled
    s_count  = pg.tooth_count
    s_depth  = pg.tooth_depth
    s_width  = pg.tooth_width
    s_taper  = pg.taper_enabled
    s_tip_sc = pg.tip_scale
    s_fn     = PROFILES.get(pg.serr_profile, serr_sine)

    # Ensure enough spanwise sections to resolve the serration teeth cleanly.
    # Each tooth needs at least SEGS_PER_TOOTH sections to look smooth.
    SEGS_PER_TOOTH = 8
    min_sections = 60
    n_sec = max(n_sec, min_sections)
    if serr_on:
        n_sec = max(n_sec, s_count * SEGS_PER_TOOTH)

    upper, lower = naca4_points(pg.naca_digits, n_pts)
    n_profile = 2 * n_pts - 2

    # Blend zone: innermost N_BLEND sections transition from hub surface to blade
    N_BLEND = max(3, n_sec // 5)

    all_verts = []
    all_faces = []

    def add_blade(az):
        blade_ring_starts = []

        for si in range(n_sec + 1):
            t_span = si / n_sec
            r      = hub_r + t_span * (tip_r - hub_r)
            chord  = chord_root + t_span * (chord_tip - chord_root)
            pitch_angle = math.atan2(pitch, 2 * math.pi * r)

            # Serration amplitude (suppressed inside the blend zone)
            if serr_on and si >= N_BLEND:
                taper_f  = (1.0 - t_span * (1.0 - s_tip_sc)) if s_taper else 1.0
                serr_amp = s_depth * taper_f
            else:
                serr_amp = 0.0

            # Full-radius airfoil section verts
            # t_serr: remap span so phase=0 at the first serrated section,
            # preventing the inboard tooth from being clipped mid-cycle.
            t_serr_start = N_BLEND / n_sec
            t_serr = max(0.0, (t_span - t_serr_start) / max(1.0 - t_serr_start, 1e-6))
            ring = section_world_verts(
                r, chord, pitch_angle, az,
                upper, lower,
                serr_amp, s_fn, s_width,
                t_serr, s_count
            )

            # ── Hub intersection ──────────────────────────────────────────
            # In the blend zone we lerp between two things:
            #   t=0 (root) : verts projected onto the hub cylinder  → wide footprint
            #   t=1 (end)  : the normal airfoil section at radius r → blade starts here
            #
            # The projection spreads the root verts around the hub in XY,
            # while preserving their Z (axial) position so the fillet height
            # follows the airfoil thickness naturally.
            if si < N_BLEND:
                t_blend   = si / N_BLEND          # 0 at hub, 1 at blend end
                hub_ring  = project_to_cylinder(ring, hub_r)
                blended   = []
                for (hx,hy,hz), (bx,by,bz) in zip(hub_ring, ring):
                    blended.append((
                        hx + t_blend*(bx - hx),
                        hy + t_blend*(by - hy),
                        hz + t_blend*(bz - hz),
                    ))
                ring = blended

            base = len(all_verts)
            blade_ring_starts.append(base)
            all_verts.extend(ring)

        # Quad strip between adjacent rings
        for si in range(n_sec):
            ba = blade_ring_starts[si]
            bb = blade_ring_starts[si + 1]
            for vi in range(n_profile):
                vn = (vi + 1) % n_profile
                all_faces.append((ba+vi, ba+vn, bb+vn, bb+vi))

        # Tip cap — fan triangles from centroid
        tip_base = blade_ring_starts[-1]
        tip_ring = list(range(tip_base, tip_base + n_profile))
        cx = sum(all_verts[i][0] for i in tip_ring) / n_profile
        cy = sum(all_verts[i][1] for i in tip_ring) / n_profile
        cz = sum(all_verts[i][2] for i in tip_ring) / n_profile
        ctr = len(all_verts)
        all_verts.append((cx, cy, cz))
        for vi in range(n_profile):
            vn = (vi + 1) % n_profile
            all_faces.append((tip_ring[vi], ctr, tip_ring[vn]))

    for bi in range(num_blades):
        add_blade(2 * math.pi * bi / num_blades)

    # ── Align blades to print bed before building hub ─────────────────────────
    # Find the lowest blade vertex and lift everything so it sits at Z = 0.
    # Then build the hub with its bottom at Z = 0 too, so both share the same floor.
    blade_min_z = min(v[2] for v in all_verts)
    all_verts = [(x, y, z - blade_min_z) for (x, y, z) in all_verts]

    # ── Hub cylinder with shaft hole ─────────────────────────────────────────
    # Outer wall + inner bore, capped with annular rings at top and bottom.
    # Bottom flush with blade bottom (Z=0); top = blade max Z + small margin.
    HUB_MARGIN = 3.0   # mm above the highest blade point
    blade_max_z = max(v[2] for v in all_verts)
    hub_segs   = 32
    z_bot = 0.0
    z_top = blade_max_z + HUB_MARGIN
    shaft_r = pg.shaft_diameter / 2.0

    # Each seg produces 4 verts: outer-bot, outer-top, inner-bot, inner-top
    hub_offset = len(all_verts)
    for seg in range(hub_segs):
        a = 2 * math.pi * seg / hub_segs
        ca, sa = math.cos(a), math.sin(a)
        all_verts.append((hub_r   * ca, hub_r   * sa, z_bot))  # 0 outer bot
        all_verts.append((hub_r   * ca, hub_r   * sa, z_top))  # 1 outer top
        all_verts.append((shaft_r * ca, shaft_r * sa, z_bot))  # 2 inner bot
        all_verts.append((shaft_r * ca, shaft_r * sa, z_top))  # 3 inner top

    for seg in range(hub_segs):
        nxt = (seg + 1) % hub_segs
        ob0 = hub_offset + seg*4 + 0;  ob1 = hub_offset + seg*4 + 1
        ib0 = hub_offset + seg*4 + 2;  ib1 = hub_offset + seg*4 + 3
        ob2 = hub_offset + nxt*4 + 0;  ob3 = hub_offset + nxt*4 + 1
        ib2 = hub_offset + nxt*4 + 2;  ib3 = hub_offset + nxt*4 + 3

        # Outer wall
        all_faces.append((ob0, ob2, ob3, ob1))
        # Inner wall (reversed so normal points inward)
        all_faces.append((ib0, ib1, ib3, ib2))
        # Bottom annular cap (outer-bot → inner-bot)
        all_faces.append((ob0, ib0, ib2, ob2))
        # Top annular cap (outer-top → inner-top)
        all_faces.append((ob1, ob3, ib3, ib1))

    # ── Assemble ─────────────────────────────────────────────────────────────

    me = bpy.data.meshes.new("Propeller")
    me.from_pydata(all_verts, [], all_faces)
    me.validate()
    me.update()
    return me


# ─────────────────────────────────────────────────────────────────────────────
#  Property Group
# ─────────────────────────────────────────────────────────────────────────────

def _upd(self, ctx): regenerate(ctx)

class PROP_PG_Settings(PropertyGroup):
    # Propeller
    diameter:    FloatProperty(name="Diameter (mm)",    default=228.6, min=50,   max=1000, step=100, precision=1, update=_upd)
    pitch:       FloatProperty(name="Pitch (mm)",        default=152.4, min=10,   max=1000, step=100, precision=1, update=_upd)
    num_blades:  IntProperty  (name="Blades",            default=2,     min=2,    max=6,               update=_upd)
    hub_radius:  FloatProperty(name="Hub Radius (mm)",   default=12.0,  min=3,    max=60,   step=10, precision=1, update=_upd)
    shaft_diameter: FloatProperty(name="Shaft Diameter (mm)", default=5.0, min=1.0, max=30.0, step=10, precision=1, update=_upd)
    # Blade
    naca_digits: IntProperty  (name="NACA 4-digit",      default=2412,  min=1,    max=9999,            update=_upd)
    chord_root:  FloatProperty(name="Chord Root (mm)",   default=30.0,  min=5,    max=200,  step=10, precision=1, update=_upd)
    chord_tip:   FloatProperty(name="Chord Tip (mm)",    default=10.0,  min=1,    max=100,  step=10, precision=1, update=_upd)
    sections:    IntProperty  (name="Span Sections",     default=30,    min=6,    max=120,             update=_upd)
    airfoil_pts: IntProperty  (name="Airfoil Points",    default=36,    min=8,    max=80,              update=_upd)
    # Serrations
    serr_enabled: BoolProperty(name="Enable Serrations", default=True,                               update=_upd)
    tooth_count:  IntProperty  (name="Tooth Count",      default=12,    min=2,    max=60,              update=_upd)
    tooth_depth:  FloatProperty(name="Tooth Depth (mm)", default=3.0,   min=0.1,  max=20,   step=10, precision=2, update=_upd)
    tooth_width:  FloatProperty(name="Valley Width",     default=0.35,  min=0.05, max=0.85, step=1,  precision=2, update=_upd)
    taper_enabled:BoolProperty (name="Tip Taper",        default=True,                               update=_upd)
    tip_scale:    FloatProperty(name="Tip Scale",        default=0.20,  min=0.0,  max=1.0,  step=1,  precision=2, update=_upd)
    serr_profile: EnumProperty (name="Tooth Profile", update=_upd, items=[
        ('SINUSOIDAL', "Sinusoidal", "Smooth sine bump — closest to owl comb"),
        ('TRIANGLE',   "Triangle",  "V-shaped teeth"),
        ('SAWTOOTH',   "Sawtooth",  "Asymmetric ramp teeth"),
    ], default='SINUSOIDAL')
    # Internal
    obj_name: StringProperty(default="")


# ─────────────────────────────────────────────────────────────────────────────
#  Regenerate
# ─────────────────────────────────────────────────────────────────────────────

def regenerate(context):
    pg  = context.scene.prop_settings
    col = bpy.data.collections.get("Propeller")
    if col is None:
        col = bpy.data.collections.new("Propeller")
        context.scene.collection.children.link(col)

    obj = bpy.data.objects.get(pg.obj_name)
    if obj is None or obj.type != 'MESH':
        me  = bpy.data.meshes.new("Propeller")
        obj = bpy.data.objects.new("Propeller", me)
        col.objects.link(obj)
        pg.obj_name = obj.name

    new_me = build_propeller(pg)
    old_me = obj.data
    obj.data = new_me
    if old_me and old_me.users == 0:
        bpy.data.meshes.remove(old_me)

    for poly in obj.data.polygons:
        poly.use_smooth = True


# ─────────────────────────────────────────────────────────────────────────────
#  Operators
# ─────────────────────────────────────────────────────────────────────────────

class PROP_OT_Generate(Operator):
    bl_idname = "prop.generate"; bl_label = "Generate Propeller"
    bl_description = "Build or rebuild the parametric propeller"
    bl_options = {'REGISTER', 'UNDO'}
    def execute(self, context):
        regenerate(context)
        self.report({'INFO'}, "Propeller generated.")
        return {'FINISHED'}



# ─────────────────────────────────────────────────────────────────────────────
#  Panel
# ─────────────────────────────────────────────────────────────────────────────

class PROP_PT_Panel(Panel):
    bl_label = "Propeller Generator"; bl_idname = "PROP_PT_panel"
    bl_space_type = 'VIEW_3D'; bl_region_type = 'UI'; bl_category = "Propeller"

    def draw(self, context):
        layout = self.layout
        pg = context.scene.prop_settings

        row = layout.row(); row.scale_y = 1.5
        row.operator("prop.generate", icon='MESH_UVSPHERE')
        layout.separator()

        b = layout.box()
        b.label(text="Propeller", icon='PROP_ON')
        b.prop(pg, "diameter"); b.prop(pg, "pitch"); b.prop(pg, "num_blades")
        b.prop(pg, "hub_radius")
        b.prop(pg, "shaft_diameter")
        layout.separator()

        b = layout.box()
        b.label(text="Blade Shape", icon='CURVE_PATH')
        b.prop(pg, "naca_digits")
        row = b.row(align=True); row.prop(pg, "chord_root"); row.prop(pg, "chord_tip")
        row = b.row(align=True); row.prop(pg, "sections");   row.prop(pg, "airfoil_pts")
        layout.separator()

        b = layout.box()
        row = b.row()
        row.label(text="Owl Serrations", icon='MOD_WAVE')
        row.prop(pg, "serr_enabled", text="")
        sub = b.column(); sub.active = pg.serr_enabled
        sub.prop(pg, "serr_profile")
        sub.prop(pg, "tooth_count")
        sub.prop(pg, "tooth_depth")
        sub.prop(pg, "tooth_width", slider=True)
        sub.separator()
        sub.prop(pg, "taper_enabled")
        r2 = sub.row(); r2.active = pg.taper_enabled
        r2.prop(pg, "tip_scale", slider=True)
        layout.separator()



# ─────────────────────────────────────────────────────────────────────────────
#  Register
# ─────────────────────────────────────────────────────────────────────────────

CLASSES = [PROP_PG_Settings, PROP_OT_Generate, PROP_PT_Panel]

def register():
    for cls in CLASSES: bpy.utils.register_class(cls)
    bpy.types.Scene.prop_settings = PointerProperty(type=PROP_PG_Settings)

def unregister():
    for cls in reversed(CLASSES): bpy.utils.unregister_class(cls)
    del bpy.types.Scene.prop_settings

if __name__ == "__main__":
    try: unregister()
    except: pass
    register()
    regenerate(bpy.context)