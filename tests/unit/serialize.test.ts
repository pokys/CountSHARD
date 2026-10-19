import { describe, expect, it } from 'vitest';
import { cs } from '../../src/i18n/cs';
import { applyGrouping, createProject, mergeDetections, removeObject } from '../../src/state/project';
import { ProjectFileError, parseProjectFile, toCsv, toProjectFile } from '../../src/io/serialize';
import { DEFAULT_GROUPING } from '../../src/model/types';

function project() {
  const p = createProject({ name: 'svazek.jpg', width: 200, height: 100, originalWidth: 4000, originalHeight: 2000, mime: 'image/jpeg' });
  const objects = mergeDetections([], [
    { x: 10, y: 10, r: 5, score: 0.91 },
    { x: 30, y: 10, r: 5.2, score: 0.8 },
    { x: 60, y: 10, r: 9, score: 0.7 },
    { x: 90, y: 10, r: 9.1, score: 0.75 },
    { x: 50, y: 50, r: 5.1, score: 0.75 },
    { x: 80, y: 50, r: 9.05, score: 0.75 },
  ]);
  return { ...p, ...applyGrouping(objects, [], DEFAULT_GROUPING) };
}

const csvOpts = {
  decimal: ',' as const,
  headers: cs.csv,
  sourceLabels: { auto: 'automaticky', manual: 'ručně' },
  reviewLabels: { unreviewed: 'nezkontrolováno', confirmed: 'potvrzeno' },
};

describe('CSV', () => {
  it('writes one row per counted object with Czech decimals and a BOM', () => {
    const p = project();
    const withRemoval = { ...p, objects: removeObject(p.objects, p.objects[0]!.id) };
    const csv = toCsv(withRemoval, csvOpts);
    expect(csv.startsWith('﻿')).toBe(true);
    const lines = csv.trim().split('\r\n');
    expect(lines).toHaveLength(1 + 5);
    expect(lines[0]).toContain('Průměr (px)');
    expect(lines[1]!.split(';')[3]).toBe('10,4'); // diameter of r = 5.2
  });

  it('adds millimetres only when calibrated', () => {
    const p = { ...project(), calibration: { mmPerPx: 2, referenceObjectId: null, referenceDiameterMm: 20 } };
    const row = toCsv(p, csvOpts).trim().split('\r\n')[1]!.split(';');
    expect(row[4]).toBe('20,0');
  });
});

describe('project file', () => {
  it('round-trips and keeps editable annotations', () => {
    const p = project();
    const file = toProjectFile(p, 'data:image/jpeg;base64,AAAA');
    const back = parseProjectFile(JSON.stringify(file));
    expect(back.project.objects).toEqual(p.objects);
    expect(back.summary.total).toBe(6);
    expect(back.summary.groups).toHaveLength(2);
  });

  it('rejects foreign or damaged files', () => {
    expect(() => parseProjectFile('nope')).toThrow(ProjectFileError);
    expect(() => parseProjectFile(JSON.stringify({ format: 'other' }))).toThrow(ProjectFileError);
    const file = toProjectFile(project(), 'data:image/jpeg;base64,AAAA');
    (file.project.objects[0] as { x: unknown }).x = 'NaN';
    expect(() => parseProjectFile(JSON.stringify(file))).toThrow(ProjectFileError);
    const noImage = { ...toProjectFile(project(), 'data:image/jpeg;base64,AAAA'), image: 'http://evil.example/x.jpg' };
    expect(() => parseProjectFile(JSON.stringify(noImage))).toThrow(ProjectFileError);
  });

  it('still opens files exported under the old app name', () => {
    const file = toProjectFile(project(), 'data:image/jpeg;base64,AAAA') as unknown as { format: string; project: { format: string } };
    file.format = 'countit-project';
    file.project.format = 'countit-project';
    expect(parseProjectFile(JSON.stringify(file)).project.format).toBe('countshard-project');
  });
});
