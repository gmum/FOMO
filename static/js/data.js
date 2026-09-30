function clip(group, method, id) {
  return 'static/videos/' + group + '/' + method + '/' + id + '.mp4';
}

function ids(count) {
  var out = [];
  for (var i = 1; i <= count; i++) out.push(i < 10 ? '0' + i : String(i));
  return out;
}

var SOTA_COLUMNS = [
  'Baseline',
  'ESD',
  'SAFREE',
  'T2VUnlearning',
  'Neg. prompt',
  { name: 'Ours', main: true }
];

var SOTA_METHODS = ['baseline', 'esd', 'safree', 't2v', 'negprompt', 'ours'];

var PAIR_COLUMNS = ['Baseline', { name: 'Ours', main: true }];

var ABLATION_COLUMNS = ['Baseline', 'L_H', 'L_V', 'L_H + L_V', { name: 'Ours', main: true }];

var ABLATION_METHODS = ['baseline', 'lh', 'lv', 'lh_lv', 'ours'];

function rowsFor(group, methods, list) {
  return list.map(function (id) {
    return { videos: methods.map(function (method) { return clip(group, method, id); }) };
  });
}

function stackedSlides(group, accent, count) {
  return ids(count).map(function (id) {
    return {
      accent: accent,
      aspect: '16 / 9',
      stacked: true,
      cols: 3,
      columns: SOTA_COLUMNS,
      rows: [{ videos: SOTA_METHODS.map(function (method) { return clip(group, method, id); }) }]
    };
  });
}

function singleSlides(group, title, accent, count) {
  return ids(count).map(function (id) {
    return {
      title: title,
      accent: accent,
      aspect: '16 / 9',
      columns: PAIR_COLUMNS,
      rows: [{ videos: [clip(group, 'baseline', id), clip(group, 'ours', id)] }]
    };
  });
}

function pairs(group, title, accent, count) {
  return {
    title: title,
    accent: accent,
    aspect: '16 / 9',
    columns: PAIR_COLUMNS,
    rows: rowsFor(group, ['baseline', 'ours'], ids(count))
  };
}

var PEOPLE = [
  ['trump', 'Donald Trump'],
  ['obama', 'Barack Obama'],
  ['lebron', 'LeBron James'],
  ['ronaldo', 'Cristiano Ronaldo'],
  ['taylor', 'Taylor Swift']
];

var MATRIX_COLUMNS = ['Original'].concat(PEOPLE.map(function (person) {
  return person[1];
}));

function matrixBlock(title, accent) {
  return {
    title: title,
    accent: accent,
    aspect: '16 / 9',
    gap: '8px',
    snug: true,
    columns: MATRIX_COLUMNS,
    rows: PEOPLE.map(function (person, index) {
      return {
        label: person[1],
        markAt: index + 1,
        videos: ['original'].concat(PEOPLE.map(function (other) {
          return other[0];
        })).map(function (column) {
          return clip('matrix', person[0], column);
        })
      };
    })
  };
}

function personSlides(accent) {
  return PEOPLE.map(function (person) {
    return {
      title: person[1],
      accent: accent,
      aspect: '16 / 9',
      narrow: true,
      columns: PAIR_COLUMNS,
      rows: rowsFor('celebrity/' + person[0], ['baseline', 'ours'], ids(3))
    };
  });
}

var NUDITY = '#d08a5a';
var VIOLENCE = '#cc7d9e';
var MOTION = '#74ab86';
var PUBLIC = '#6f93cc';
var ABLATION = '#8896a6';

window.PAGE_DATA = {
  sections: [
    {
      id: 'ablation',
      heading: 'Loss ablation',
      accent: ABLATION,
      layout: 'grid',
      blocks: [{
        accent: ABLATION,
        aspect: '16 / 9',
        columns: ABLATION_COLUMNS,
        rows: rowsFor('ablation', ABLATION_METHODS, ids(4))
      }]
    },
    {
      id: 'nudity',
      heading: 'Nudity',
      accent: NUDITY,
      layout: 'carousel',
      blocks: stackedSlides('nudity', NUDITY, 13)
    },
    {
      id: 'general-safety',
      heading: 'General safety',
      accent: VIOLENCE,
      layout: 'carousels',
      blocks: [
        singleSlides('gore', 'Gore', VIOLENCE, 9),
        singleSlides('violence', 'Violence', VIOLENCE, 8)
      ]
    },
    {
      id: 'motion',
      heading: 'Motion',
      accent: MOTION,
      layout: 'grid',
      blocks: [
        pairs('knife', 'Brandishing a knife', MOTION, 3),
        pairs('fighting', 'Fighting', MOTION, 4),
        pairs('jumping', 'Jumping', MOTION, 3),
        pairs('running', 'Running', MOTION, 4),
        pairs('tongue', 'Sticking out the tongue', MOTION, 2),
        pairs('dancing', 'Dancing', MOTION, 2)
      ]
    },
    {
      id: 'public-figures',
      heading: 'Public figures',
      accent: PUBLIC,
      layout: 'mixed',
      blocks: [
        matrixBlock('Identity matrix', PUBLIC),
        personSlides(PUBLIC)
      ]
    }
  ]
};
