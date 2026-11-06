"""VisIt 3.5 CLI renderer for specfem_visit.py (English-only scripting).

This file is meant to be launched by VisIt's Python interpreter, not by
plain python3. The CLI must be given SPECFEM_VISIT_CONFIG in the environment.
"""
import json
import os
import sys


def fail(message):
    try:
        detail = GetLastError()
        if detail:
            message += ' | VisIt: ' + str(detail)
    except Exception:
        pass
    raise RuntimeError(message)


def require(result, operation):
    if result == 0:
        fail('%s failed' % operation)


def register_color_table(name, colors):
    """Define reproducible, perceptually ordered colors in VisIt itself."""
    try:
        table = ColorControlPointList()
        for position, rgb in colors:
            point = ColorControlPoint()
            point.position = float(position)
            point.colors = tuple(int(component) for component in rgb) + (255,)
            table.AddControlPoints(point)
        result = AddColorTable(name, table)
        if result == 0:
            print('[WARNING] Could not register color table %s' % name, flush=True)
            return False
        print('[COLOR] Custom table:', name, flush=True)
        return True
    except Exception as exc:
        print('[WARNING] Could not configure %s: %s' % (name, exc), flush=True)
        return False


def scalar_field(domain, field):
    if field == 'potential' and domain == 'acoustic':
        return 'potential'
    if field == 'magnitude':
        return 'displacement_magnitude'
    if field in ('x', 'y'):
        component = 0 if field == 'x' else 1
        name = 'sf_%s_%s' % (domain, field)
        require(DefineScalarExpression(name, '<displacement>[%d]' % component),
                'DefineScalarExpression(%s)' % name)
        return name
    fail('Unsupported field: %s in %s' % (field, domain))


def set_plot_colors(plot_index, field, limit, domain, palettes):
    require(SetActivePlots(plot_index), 'SetActivePlots(%d)' % plot_index)
    attributes = PseudocolorAttributes()
    attributes.scaling = attributes.Linear
    attributes.lightingFlag = 0
    if limit is None:
        fail('Missing fixed color limit for %s. Run specfem_visit.py again.' % domain)
    if limit is not None:
        attributes.minFlag = 1
        attributes.maxFlag = 1
        attributes.min = 0.0 if field == 'magnitude' else -float(limit)
        attributes.max = float(limit)
    palette = palettes.get('solid' if domain == 'viscoelastic' else 'water')
    if palette:
        attributes.colorTableName = palette
    else:
        # Use conservative built-in fallbacks if custom table registration fails.
        try:
            available = tuple(GetColorTableNames())
            choices = (('coolwarm', 'RdBu', 'hot_and_cold') if domain == 'viscoelastic'
                       else ('viridis', 'magma', 'hot'))
            for candidate in choices:
                if candidate in available:
                    attributes.colorTableName = candidate
                    break
        except Exception:
            pass
    require(SetPlotOptions(attributes), 'SetPlotOptions(%s)' % domain)


def prepare_legend(index, domain, field, count):
    """Keep compact vertical legends entirely within the left-hand margin.

    Legend positions are screen-normalized coordinates. VisIt places the
    vertical color bar below its position anchor, so placing the lower
    legend too close to y=0 clips it from the saved image.
    """
    try:
        plot_name = GetPlotList().GetPlots(index).plotName
        legend = GetAnnotationObject(plot_name)
        legend.managePosition = 0
        legend.drawMinMax = 0
        legend.drawTitle = 1
        legend.useCustomTitle = 1
        legend.numberFormat = '%0.2g'

        # Keep tick labels on the right and away from the vertical axis title.
        # VisIt defaults to vertical legends with labels on the right.
        legend.xScale = 1.0

        if count == 2:
            # A 1920x1080 canvas has room for both bars with these positions.
            # ~280-pixel-high bars with a comfortable gap between them.
            legend.position = ((0.023, 0.81) if index == 0
                               else (0.023, 0.47))
            legend.yScale = 1.15
        else:
            legend.position = (0.023, 0.69)
            legend.yScale = 1.40

        # Titles intentionally short: the full physical fields are noted in
        # plot configuration, not in a long legend heading across the plot.
        if domain == 'viscoelastic':
            legend.customTitle = ('Sediment |u|' if field == 'magnitude'
                                  else 'Sediment u' + field)
        else:
            legend.customTitle = ('Water potential' if field == 'potential'
                                  else ('Water |u|' if field == 'magnitude'
                                        else 'Water u' + field))

        print('[LEGEND] %s at %s, yScale=%.2f, title=%s' % (
            domain, legend.position, legend.yScale, legend.customTitle),
            flush=True)
    except Exception as exc:
        print('[WARNING] Could not position %s legend: %s' % (
            domain, exc), flush=True)


def configure_axes():
    """English axes and a clean white background."""
    try:
        annotations = GetAnnotationAttributes()
        annotations.databaseInfoFlag = 0
        annotations.userInfoFlag = 0
        annotations.axes2D.visible = 1
        annotations.axes2D.xAxis.title.userTitle = 1
        annotations.axes2D.xAxis.title.title = 'Horizontal distance'
        annotations.axes2D.xAxis.title.userUnits = 1
        annotations.axes2D.xAxis.title.units = 'm'
        annotations.axes2D.yAxis.title.userTitle = 1
        annotations.axes2D.yAxis.title.title = 'Vertical coordinate'
        annotations.axes2D.yAxis.title.userUnits = 1
        annotations.axes2D.yAxis.title.units = 'm'
        # Keep axes and labels visible on the enlarged figure.
        require(SetAnnotationAttributes(annotations), 'SetAnnotationAttributes')
    except Exception as exc:
        print('[WARNING] Axis annotation settings not applied: %s' % exc, flush=True)


def make_text_annotation(text, position, height, bold):
    """Create the title/metadata; unsupported cosmetic options are optional."""
    try:
        label = CreateAnnotationObject('Text2D')
        label.text = str(text)
        label.position = tuple(position)
        label.height = float(height)
        label.fontBold = 1 if bold else 0
        return label
    except Exception as exc:
        print('[WARNING] Text annotation unavailable: %s' % exc, flush=True)
        return None


def update_view(window):
    view = View2DAttributes()
    view.windowCoords = tuple(window)
    # Wide drawing viewport: both long color legends live in the left margin.
    view.viewportCoords = (0.17, 0.97, 0.14, 0.89)
    view.fullFrameActivationMode = view.On
    view.windowValid = 1
    require(SetView2D(view), 'SetView2D')


config_file = os.environ.get('SPECFEM_VISIT_CONFIG')
if not config_file:
    raise RuntimeError('SPECFEM_VISIT_CONFIG missing: launch specfem_visit.py JOB_ID')
with open(config_file, encoding='utf-8') as stream:
    cfg = json.load(stream)

datasets = cfg['datasets']
order = [domain for domain in ('viscoelastic', 'acoustic') if domain in datasets]
frames = cfg['frames']
first_state = frames[0]
views = cfg['views']
for settings in views.values():
    os.makedirs(settings['output_dir'], exist_ok=True)

# V5: swap the two V4 color schemes as requested.
# Water now uses the former sediment sequential palette; sediment uses
# the former water blue-neutral-coral diverging palette.
# This changes visualization only, never the exported physical data.
palettes = {}
if register_color_table('SPECFEM_Sediment_2026', [
    (0.00, (27, 43, 88)),
    (0.18, (49, 93, 145)),
    (0.42, (50, 147, 161)),
    (0.70, (154, 198, 152)),
    (0.87, (240, 211, 137)),
    (1.00, (199, 86, 63)),
]):
    palettes['water'] = 'SPECFEM_Sediment_2026'
if register_color_table('SPECFEM_Water_2026', [
    (0.00, (40, 79, 136)),
    (0.25, (103, 159, 186)),
    (0.50, (243, 242, 229)),
    (0.75, (225, 165, 108)),
    (1.00, (163, 70, 65)),
]):
    palettes['solid'] = 'SPECFEM_Water_2026'

if len(order) > 1:
    require(SetDatabaseCorrelationOptions(0, 1), 'SetDatabaseCorrelationOptions')
try:
    SetPipelineCachingMode(0)
except Exception:
    pass

for index, domain in enumerate(order):
    dataset = datasets[domain]
    print('[VISIT] Opening %s: %s' % (domain, dataset['database']), flush=True)
    require(OpenDatabase(dataset['database'], first_state), 'OpenDatabase(%s)' % domain)
    variable = scalar_field(domain, dataset['field'])
    require(AddPlot('Pseudocolor', variable), 'AddPlot(%s)' % domain)
    # Critical for VisIt 3.5: the initial DrawPlots must come BEFORE plot options.
    require(DrawPlots(), 'DrawPlots(%s)' % domain)
    set_plot_colors(index, dataset['field'], dataset['limit'], domain, palettes)

if len(order) > 1:
    databases = tuple(datasets[domain]['database'] for domain in order)
    correlation = 'SPECFEM_SYNCHRONIZED'
    require(CreateDatabaseCorrelation(correlation, databases, 0), 'CreateDatabaseCorrelation')
    require(SetActiveTimeSlider(correlation), 'SetActiveTimeSlider')

states = TimeSliderGetNStates()
if states != len(cfg['times']):
    fail('VisIt has %d states; config specifies %d' % (states, len(cfg['times'])))
require(SetTimeSliderState(first_state), 'SetTimeSliderState')

if 'global' in views and views['global']['window'] is None:
    # ResetView encompasses the joint extent of water and sediment plots.
    ResetView()
    automatic_bounds = tuple(GetView2D().windowCoords)
    if len(automatic_bounds) != 4 or not (
            automatic_bounds[0] < automatic_bounds[1]
            and automatic_bounds[2] < automatic_bounds[3]):
        fail('Invalid automatically computed global bounds: %s' % (automatic_bounds,))
    views['global']['window'] = list(automatic_bounds)
    print('[VIEW] Automatic global bounds:', automatic_bounds, flush=True)

configure_axes()
for index, domain in enumerate(order):
    prepare_legend(index, domain, datasets[domain]['field'], len(order))

title = make_text_annotation('', (0.19, 0.945), 0.032, True)
metadata = make_text_annotation('', (0.76, 0.942), 0.023, False)

save = SaveWindowAttributes()
save.outputToCurrentDirectory = 0
save.family = 0
save.format = save.PNG
save.width = int(cfg['width'])
save.height = int(cfg['height'])
save.resConstraint = save.NoConstraint

for ordinal, state in enumerate(frames):
    if state != first_state:
        require(SetTimeSliderState(state), 'SetTimeSliderState(%d)' % state)
    seconds = cfg['times'][state]
    for view_name, settings in views.items():
        update_view(settings['window'])
        if title is not None:
            title.text = '%s | %s view' % (cfg.get('title', 'Acoustic-elastic waves'),
                                          view_name.capitalize())
        if metadata is not None:
            metadata.text = 'Job %s  |  t = %.3f s  |  %d/%d' % (
                cfg['job_id'], seconds, state + 1, len(cfg['times']))
            if cfg.get('scale_percentile', 100.0) < 100.0:
                metadata.text += '  |  Color-clipped p%g' % cfg['scale_percentile']
        save.outputDirectory = settings['output_dir']
        save.fileName = 'frame_%04d' % ordinal
        require(SetSaveWindowAttributes(save), 'SetSaveWindowAttributes')
        image_name = SaveWindow()
        print('[PNG] %s %02d/%02d  t=%0.3fs -> %s' % (
            view_name, ordinal + 1, len(frames), seconds, image_name), flush=True)

print('[COMPLETE] Rendered views: %s' % ', '.join(views), flush=True)
# No Exit() call: this function is not defined in VisIt CLI 3.5.
sys.exit(0)
