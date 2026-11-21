"""Create the Part 1 PDF and vector figures from the archived measurements.

Optional report dependencies: numpy and reportlab==4.4.9.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.units import inch
from reportlab.graphics.shapes import Drawing, String, Line
from reportlab.graphics.charts.lineplots import LinePlot
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Preformatted

ROOT = Path(__file__).resolve().parents[1]
BLUE, ORANGE = colors.HexColor('#2468A2'), colors.HexColor('#BD5B22')
BRANCH = 'https://github.com/prathampatel21/cybershuttle-ai4science-foundry/tree/assignment5-chgnet-eos/experiments/chgnet-eos'


def figure(rows, summary, kind):
    drawing = Drawing(480, 200)
    for col, name in enumerate(('Si', 'LiMnO2')):
        dx = col * 245
        lp = LinePlot(); lp.x = dx + 40; lp.y = 35; lp.width = 185; lp.height = 125
        lp.lines[0].strokeColor = BLUE; lp.lines[0].strokeWidth = 1.5
        lp.lines[1].strokeColor = ORANGE; lp.lines[1].strokeWidth = 1.5; lp.lines[1].strokeDashArray = [4, 2]
        if kind == 'energy':
            minimum = min(r['energy_eV_atom'] for r in rows if r['material'] == name)
            lp.data = [[(r['volume_factor'], 1000 * (r['energy_eV_atom'] - minimum))
                        for r in rows if r['material'] == name and r['mode'] == mode]
                       for mode in ('frozen', 'relaxed')]
            lp.xValueAxis.valueMin = .88; lp.xValueAxis.valueMax = 1.12; lp.xValueAxis.valueStep = .06
            lp.xValueAxis.labelTextFormat = '%.2f'
            lp.xValueAxis.valueSteps = [.88, .94, 1.00, 1.06, 1.12]
            xlabel, ylabel = 'Volume / initial volume', 'Energy above sampled minimum (meV/atom)'
        else:
            lp.data = [[(r['n_points'], r['stiffness_GPa']) for r in summary['fits']
                        if r['material'] == name and r['mode'] == mode and r['valid']]
                       for mode in ('frozen', 'relaxed')]
            lp.xValueAxis.valueMin = 5; lp.xValueAxis.valueMax = 13; lp.xValueAxis.valueStep = 4
            lp.xValueAxis.valueSteps = [5, 9, 13]
            xlabel, ylabel = 'Points in quadratic fit', 'Effective isotropic stiffness (GPa)'
        lp.xValueAxis.labels.fontSize = lp.yValueAxis.labels.fontSize = 7
        lp.yValueAxis.labelTextFormat = '%.0f'
        drawing.add(lp)
        drawing.add(String(dx + 132, 177, name, fontName='Helvetica-Bold', fontSize=11, textAnchor='middle'))
        drawing.add(String(dx + 132, 12, xlabel, fontSize=8, textAnchor='middle'))
        drawing.add(String(dx + 5, 165, ylabel, fontSize=6.4))
    drawing.add(Line(140, 192, 156, 192, strokeColor=BLUE, strokeWidth=1.5))
    drawing.add(String(160, 189, 'Frozen sites', fontSize=8))
    drawing.add(Line(240, 192, 256, 192, strokeColor=ORANGE, strokeWidth=1.5, strokeDashArray=[4, 2]))
    drawing.add(String(260, 189, 'Relaxed sites', fontSize=8))
    return drawing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--run', type=Path, default=ROOT / 'evidence/7229001')
    args = ap.parse_args()
    run = args.run
    rows = json.loads((run / 'scan.json').read_text())
    s = json.loads((run / 'summary.json').read_text())
    meta = json.loads((run / 'metadata.json').read_text())
    assert len(rows) == 68 and all(r.get('converged', True) for r in rows)
    assert (run / 'reference-validation.exit').read_text().strip() == '0'
    fits = {(r['material'], r['mode']): r for r in s['fits'] if r['half_width'] == 2}
    decrease = 100 * (1 - fits['LiMnO2', 'relaxed']['stiffness_GPa'] / fits['LiMnO2', 'frozen']['stiffness_GPa'])
    tight = next(r for r in s['refinement'] if r['material'] == 'LiMnO2')
    force_sensitivity = 100 * (tight['tight_fit']['stiffness_GPa'] / fits['LiMnO2', 'relaxed']['stiffness_GPa'] - 1)
    # Independent analysis of the same stored predictions, not a new GPU run.
    pressure_slopes = []
    for name in ('Si', 'LiMnO2'):
        for mode in ('frozen', 'relaxed'):
            data = [r for r in rows if r['material'] == name and r['mode'] == mode]
            i = int(np.argmin([r['energy_eV_atom'] for r in data]))
            sample = data[i-2:i+3]
            slope, _ = np.polyfit([r['volume_A3_atom'] for r in sample], [r['pressure_GPa'] for r in sample], 1)
            pressure_slopes.append({'material': name, 'mode': mode,
                                   'stiffness_GPa': float(-fits[name, mode]['volume_A3_atom'] * slope)})
    (run / 'pressure-slope-analysis.json').write_text(json.dumps(pressure_slopes, indent=2) + '\n')
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='BodyReport', fontName='Helvetica', fontSize=10, leading=13, spaceAfter=7))
    styles.add(ParagraphStyle(name='SmallReport', fontName='Helvetica', fontSize=8, leading=10, spaceAfter=5))
    styles['Title'].fontName='Helvetica-Bold'; styles['Title'].fontSize=21; styles['Title'].leading=25; styles['Title'].alignment=0
    styles['Heading1'].fontName='Helvetica-Bold'; styles['Heading1'].fontSize=12; styles['Heading1'].leading=16; styles['Heading1'].spaceBefore=8
    story=[]
    def p(text): story.append(Paragraph(text, styles['BodyReport']))
    def small(text): story.append(Paragraph(text, styles['SmallReport']))
    def h(text): story.append(Paragraph(text, styles['Heading1']))
    story.append(Paragraph('Atomic relaxation and CHGNet volume response', styles['Title']))
    p('Pratham Patel | Assignment 5 Part 1 | November 21, 2026')
    p(f'<b>Finding.</b> In this CHGNet experiment, allowing atomic positions to relax lowers the local effective isotropic stiffness of LiMnO2 by {decrease:.1f}%, while ideal diamond silicon is unchanged. All 34 primary relaxations converged. The result demonstrates a mechanical-response application of the existing inference workflow; it does not establish accuracy against DFT or experiment.')
    small(f'<a href="{BRANCH}" color="#145A93">Experimental code and README</a> | <a href="{BRANCH}/evidence/7229001" color="#145A93">Raw execution evidence</a> | <a href="https://github.com/cyber-shuttle/cybershuttle-ai4science-foundry/pull/39" color="#145A93">Original Foundry contribution</a>')
    h('Research objective and technical approach')
    p('How much does internal atomic relaxation change CHGNet energy-volume curves, and how sensitive is the derived stiffness to numerical choices? This matters when using a pretrained interatomic potential to screen mechanical response: a frozen structure may appear stiffer than one whose atoms can rearrange. Silicon provides a simple symmetry control, while the LiMnO2 example extends the battery-material system used in my Foundry contribution.')
    p('I combined CHGNet checkpoint 0.3.0 with its ASE calculator, FIRE position optimization, controlled lattice scaling, and an analysis layer for pressure and curvature. The main challenges were energy-per-atom versus total-energy units, stress sign conventions, force convergence, and estimating second derivatives from finite samples. This is an adaptation to a new property-estimation task, not another reproduction of a single inference call.')
    h('Experimental design')
    p('Both cells contain eight atoms. Silicon is generated as ideal diamond with an initial lattice parameter of 5.43 Angstrom; that value is an initialization, not a reference measurement. LiMnO2 is the upstream mp-18767 example. For each material I tested 17 volume factors from 0.88 to 1.12 in increments of 0.015. Cell vectors scale by the cube root of that factor. Each volume starts independently from the original fractional coordinates.')
    p('At every volume I evaluated frozen sites, then relaxed positions at fixed cell with FIRE to a maximum force of 0.03 eV/Angstrom, allowing 200 steps. Five points near each sampled minimum were refined to 0.01 eV/Angstrom. Quadratic fits use 5, 9 or 13 points around the sampled minimum; edge minima and nonpositive curvature are rejected. I calculate B_path = V0 d2E/dV2, with E and V both per atom and 1 eV/Angstrom3 = 160.21766208 GPa.')
    p('<b>Interpretation.</b> Cell shape stays fixed. For anisotropic LiMnO2 this is a constrained effective isotropic stiffness, not a fully hydrostatic bulk modulus obtained after cell-shape relaxation.')
    h('Local five point fit results')
    data=[['Material and treatment','V0 (Angstrom3/atom)','B_path (GPa)']]
    for name in ('Si','LiMnO2'):
        for mode in ('frozen','relaxed'):
            r=fits[name,mode];data.append([name+' '+mode,f"{r['volume_A3_atom']:.4f}",f"{r['stiffness_GPa']:.2f}"])
    t=Table(data,colWidths=[185,160,135],hAlign='LEFT')
    t.setStyle(TableStyle([('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),('FONTSIZE',(0,0),(-1,-1),9),('BACKGROUND',(0,0),(-1,0),colors.HexColor('#EDF1F5')),('GRID',(0,0),(-1,-1),.4,colors.HexColor('#CDD4DB')),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6),('ALIGN',(1,1),(-1,-1),'CENTER')]))
    story.append(t)
    story.append(PageBreak())
    h('Measured curves and numerical sensitivity')
    story.append(figure(rows,s,'energy'))
    small('Figure 1. Energy relative to the lowest sampled energy for each material. Silicon curves coincide. LiMnO2 gains energy by relaxing, especially under compression; these are model predictions for the same composition, not independent reference energies.')
    story.append(figure(rows,s,'stiffness'))
    small('Figure 2. Effective stiffness changes as the quadratic fitting window expands. Frozen and relaxed silicon overlap; LiMnO2 remains softer after relaxation at all three widths. Window variation is a numerical sensitivity check, not a statistical confidence interval.')
    h('Observations')
    p('LiMnO2 relaxation lowers energy by 0.16 to 26.13 meV/atom across the scan and takes at most 33 optimization steps. Its five-point estimate falls from 125.90 to 87.59 GPa. Silicon requires zero position updates at every volume, so frozen and relaxed energies coincide and both give 85.96 GPa locally. This control supports attributing the LiMnO2 change to internal position optimization within the chosen model and constraints.')
    p(f'Tightening the LiMnO2 force threshold gives {tight["tight_fit"]["stiffness_GPa"]:.2f} GPa, a {force_sensitivity:.2f}% increase, with at most 0.072 meV/atom additional energy lowering. Across the 5/9/13-point fits, frozen LiMnO2 spans 125.90-144.22 GPa and relaxed LiMnO2 87.59-92.98 GPa; silicon spans 79.98-85.96 GPa. The relaxation trend is stable, but a single fitted modulus should not be treated as exact.')
    p('At three volumes per branch, central energy differences with relative volume steps 0.001 and 0.005 agree with pressure from -trace(stress)/3. Across 24 checks, the largest absolute discrepancy is 0.0676 GPa (rounded upward). Using h=0.001, the largest is 0.0132 GPa. Stress is differentiated from the same model energy, so this tests implementation and numerical consistency rather than physical accuracy.')
    story.append(PageBreak())
    h('Environment and reproduction')
    p(f'The measured run used ICE job 7229001, account coc, partition ice-gpu: one Tesla V100-PCIE-16GB, four CPU cores, 16 GB requested RAM and a one-hour allocation. Python {meta["python"]}, CHGNet package {meta["packages"]["chgnet"]}, checkpoint 0.3.0, PyTorch {meta["packages"]["torch"]}, NumPy {meta["packages"]["numpy"]}, ASE {meta["packages"]["ase"]} and pymatgen {meta["packages"]["pymatgen"]} are recorded in metadata.json and the package lock. Seed is 42; torch threads are four.')
    p(f'The scan/refinement loop took {meta["elapsed_seconds"]:.2f} seconds, excluding imports, model loading, setup and the separate reference validation. Peak PyTorch allocated GPU memory was {meta["gpu_peak_allocated_MiB"]:.1f} MiB; this excludes other GPU allocations. The GPU was released after collecting evidence.')
    p('The README gives a fresh scratch clone and locked environment setup. For this run I reused the existing validated CHGNet environment and pinned source, rather than reinstalling. Home storage was at quota, so I obtained the allocation directly with srun and wrote experiment outputs to scratch. No existing files were deleted. Model/input checksums and a fresh reference validation support traceability; a fresh installation of the new experiment itself remains a future check.')
    cmd='''# In an allocated GPU shell; follow README for a fresh setup.
cd "$HOME/scratch/chgnet-eos/experiments/chgnet-eos"
WORKFLOW_ROOT="$HOME/chgnet-inference" DEVICE=cuda bash scripts/run.sh
# Default output uses the Slurm ID; CPU requires DEVICE=cpu.
python scripts/analyze.py runs/7229001'''
    story.append(Preformatted(cmd,ParagraphStyle('Commands',fontName='Courier',fontSize=7.6,leading=10,spaceAfter=9)))
    h('Execution evidence and code traceability')
    p('The archived execution log reports no broken dependencies, two passing analysis tests and exit status 0. scan.json stores 68 evaluated states; all 17 relaxations per material and all 10 force-threshold refinements converged. Per-volume FIRE logs and CIF files preserve the trajectories of the investigation. summary.json, scan.csv and derivative-checks.json contain the reported results.')
    p('A fresh run of the original reference and symmetry checks on this V100 also passed all four numerical comparisons and all five transformations. The scheduler snapshot was collected while the allocation was active; its RUNNING state is not the experiment outcome. The separate experiment and reference exit files both contain 0.')
    small('Executed repository commit: e26db6602f906a500aebca6276cbf23e232911ed.<br/>Pinned upstream source: b9a6c15860237297a5a5169660e83b2e61a78e76.<br/>The experiment script hash is stored in metadata.json and matches the published source. Individual implementation: volume-scan runner, ASE integration, sensitivity analysis, tests and evidence documentation.')
    h('Limitations and next steps')
    p('Only two small cells and one checkpoint were tested; training overlap is unknown. There are no new DFT labels, finite-temperature effects, cell-shape relaxations or calibrated uncertainty estimates. A quadratic approximation over a broad interval mixes local curvature with anharmonic response, explaining why window sensitivity must be reported. Small force residuals can still affect second derivatives.')
    p('For Assignment 6 I would densify the minimum region, compare quadratic and Birch-Murnaghan fits, relax cell shape, and verify convergence with tighter forces. Then I would compare against a provenance-checked DFT equation of state or a second potential using identical structures, units and constraints. A fresh scratch installation would test the new experiment end to end.')
    small('<b>Scientific sources:</b> <a href="https://chgnet.lbl.gov/" color="#145A93">CHGNet model and applications</a>; <a href="https://github.com/CederGroupHub/chgnet/blob/b9a6c15860237297a5a5169660e83b2e61a78e76/chgnet/model/model.py" color="#145A93">pinned energy and stress implementation</a>; <a href="https://github.com/CederGroupHub/chgnet/blob/b9a6c15860237297a5a5169660e83b2e61a78e76/chgnet/model/dynamics.py" color="#145A93">pinned ASE interface</a>. This document covers Part 1 only; the presentation recording is outside the current scope.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def footer(canvas, doc):
        canvas.setFont('Helvetica',8);canvas.setFillColor(colors.HexColor('#55616E'))
        canvas.drawString(54,30,'CHGNet research proof of concept | Pratham Patel')
        canvas.drawRightString(558,30,str(doc.page))
    doc=SimpleDocTemplate(str(args.output),pagesize=(612,792),rightMargin=54,leftMargin=54,topMargin=45,bottomMargin=45,title='Atomic relaxation and CHGNet volume response',author='Pratham Patel',subject='Assignment 5 Part 1 research proof of concept')
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    print(args.output)


if __name__ == '__main__':
    main()
