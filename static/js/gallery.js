(function () {
  'use strict';

  var DEADBAND = 0.02;
  var MAX_SOFT = 0.35;
  var GAIN = 1.6;
  var RATE_LIMIT = 0.14;
  var CHECK_EVERY = 250;
  var LOAD_TIMEOUT = 12000;

  var PREV = '<svg viewBox="0 0 24 24"><path d="M15.2 3.8 6.9 12l8.3 8.2 2.1-2.1L11.1 12l6.2-6.1z"/></svg>';
  var NEXT = '<svg viewBox="0 0 24 24"><path d="M8.8 3.8 6.7 5.9 12.9 12l-6.2 6.1 2.1 2.1L17.1 12z"/></svg>';

  function make(tag, className, html) {
    var node = document.createElement(tag);
    if (className) node.className = className;
    if (html != null) node.innerHTML = html;
    return node;
  }

  function makeText(tag, className, text) {
    var node = document.createElement(tag);
    if (className) node.className = className;
    if (text != null) node.textContent = text;
    return node;
  }

  function Group(root) {
    this.root = root;
    this.videos = Array.prototype.slice.call(root.querySelectorAll('video'));
    this.visible = false;
    this.selected = true;
    this.running = false;
    this.loaded = false;
    this.ready = false;
    this.duration = 0;
    this.useNativeLoop = true;
    this.lastCheck = 0;
    this.frame = null;
    this.videos.forEach(function (video) {
      video.muted = true;
      video.defaultMuted = true;
      video.playsInline = true;
      video.setAttribute('playsinline', '');
      video.setAttribute('webkit-playsinline', '');
      video.setAttribute('disablepictureinpicture', '');
      video.setAttribute('disableremoteplayback', '');
    });
  }

  Group.prototype.first = function () {
    return this.videos[0];
  };

  Group.prototype.load = function () {
    if (this.loaded) return;
    this.loaded = true;
    this.videos.forEach(function (video) {
      if (!video.getAttribute('src') && video.dataset.src) {
        video.setAttribute('src', video.dataset.src);
        video.load();
      }
    });
  };

  Group.prototype.unload = function () {
    if (!this.loaded) return;
    this.loaded = false;
    this.ready = false;
    this.root.classList.remove('ready');
    this.videos.forEach(function (video) {
      video.pause();
      video.removeAttribute('src');
      video.load();
    });
  };

  Group.prototype.waitForVideos = function () {
    if (this.pending) return this.pending;
    this.pending = Promise.all(this.videos.map(function (video) {
      return new Promise(function (resolve) {
        if (video.readyState >= 3) return resolve();
        var settled = false;
        function done() {
          if (settled) return;
          settled = true;
          clearTimeout(timer);
          video.removeEventListener('canplay', done);
          video.removeEventListener('loadeddata', done);
          video.removeEventListener('error', done);
          resolve();
        }
        var timer = setTimeout(done, LOAD_TIMEOUT);
        video.addEventListener('canplay', done);
        video.addEventListener('loadeddata', done);
        video.addEventListener('error', done);
      });
    }));
    return this.pending;
  };

  Group.prototype.measure = function () {
    var lengths = this.videos.map(function (video) {
      return video.duration;
    }).filter(function (value) {
      return isFinite(value) && value > 0.05;
    });
    if (!lengths.length) {
      this.duration = 0;
      return;
    }
    var shortest = Math.min.apply(null, lengths);
    var longest = Math.max.apply(null, lengths);
    this.duration = shortest;
    this.useNativeLoop = (longest - shortest) < 0.06 && lengths.length === this.videos.length;
    var native = this.useNativeLoop;
    this.videos.forEach(function (video) {
      video.loop = native;
    });
  };

  Group.prototype.refresh = function () {
    if (this.visible && this.selected) this.start();
    else this.stop();
  };

  Group.prototype.start = function () {
    if (this.running) return;
    this.running = true;
    this.load();
    var group = this;
    this.waitForVideos().then(function () {
      if (!group.running) return;
      if (!group.ready) {
        group.ready = true;
        group.measure();
        group.root.classList.add('ready');
      }
      group.restart();
      group.step();
    });
  };

  Group.prototype.stop = function () {
    this.running = false;
    if (this.frame) cancelAnimationFrame(this.frame);
    this.frame = null;
    this.videos.forEach(function (video) {
      video.pause();
    });
  };

  Group.prototype.play = function () {
    this.videos.forEach(function (video) {
      var started = video.play();
      if (started && started.catch) started.catch(function () {});
    });
  };

  Group.prototype.restart = function () {
    this.videos.forEach(function (video) {
      video.playbackRate = 1;
      try { video.currentTime = 0; } catch (error) {}
    });
    this.play();
  };

  Group.prototype.step = function () {
    if (!this.running) return;
    this.sync();
    var group = this;
    this.frame = requestAnimationFrame(function () {
      group.step();
    });
  };

  Group.prototype.sync = function () {
    var lead = this.first();
    if (!lead || !this.duration) return;

    var time = lead.currentTime;

    if (!this.useNativeLoop && time >= this.duration - 0.035) {
      this.restart();
      return;
    }

    var now = performance.now();
    if (now - this.lastCheck < CHECK_EVERY) return;
    this.lastCheck = now;

    for (var i = 1; i < this.videos.length; i++) {
      var video = this.videos[i];
      if (video.seeking || video.readyState < 2) continue;

      var offset = video.currentTime - time;
      if (this.useNativeLoop) {
        if (offset > this.duration / 2) offset -= this.duration;
        if (offset < -this.duration / 2) offset += this.duration;
      }
      var size = Math.abs(offset);

      if (size > MAX_SOFT) {
        video.playbackRate = 1;
        try { video.currentTime = time; } catch (error) {}
      } else if (size > DEADBAND) {
        var rate = 1 - offset * GAIN;
        rate = Math.max(1 - RATE_LIMIT, Math.min(1 + RATE_LIMIT, rate));
        if (Math.abs(video.playbackRate - rate) > 0.005) video.playbackRate = rate;
      } else if (video.playbackRate !== 1) {
        video.playbackRate = 1;
      }

      if (video.paused) {
        var resumed = video.play();
        if (resumed && resumed.catch) resumed.catch(function () {});
      }
    }

    if (lead.paused) {
      var started = lead.play();
      if (started && started.catch) started.catch(function () {});
    }
  };

  function readColumn(value) {
    return typeof value === 'string' ? { name: value } : (value || {});
  }

  function buildTile(column, source, aspect, mark) {
    var info = readColumn(column);
    var classes = 'tile';
    if (info.main) classes += ' main';
    if (mark) classes += ' mark';
    if (info.split) classes += ' split';

    var tile = make('figure', classes);
    tile.appendChild(makeText('figcaption', 'tile-label', info.name || ''));

    var frame = make('div', 'frame');
    frame.style.setProperty('--aspect', aspect || '16 / 9');

    if (source) {
      var video = document.createElement('video');
      video.dataset.src = source;
      video.preload = 'none';
      frame.appendChild(video);
    }

    tile.appendChild(frame);
    return tile;
  }

  function buildHead(block) {
    var head = make('div', 'block-head');
    if (block.title) head.appendChild(makeText('h3', null, block.title));
    if (block.description) head.appendChild(makeText('p', null, block.description));
    return head;
  }

  function buildBlock(block, skipHead) {
    var root = make('div', 'block group' + (block.stacked ? ' stacked' : ''));
    if (block.accent) root.style.setProperty('--accent', block.accent);
    if (block.gap) root.style.setProperty('--gap', block.gap);

    if (!skipHead && (block.title || block.description)) {
      root.appendChild(buildHead(block));
    }

    var columns = block.columns || [];
    var panel = make('div', block.narrow ? 'panel narrow' : 'panel');

    var heads = make('div', 'heads');
    heads.style.setProperty('--cols', Math.max(1, columns.length));
    columns.forEach(function (column) {
      var info = readColumn(column);
      var classes = '';
      if (info.main) classes += 'main ';
      if (info.split) classes += 'split';
      heads.appendChild(makeText('span', classes.trim() || null, info.name || ''));
    });
    panel.appendChild(heads);

    var hasLabels = (block.rows || []).some(function (row) {
      return !!row.label;
    });
    var rows = make('div', hasLabels ? (block.snug ? 'rows snug' : 'rows') : 'rows tight');

    (block.rows || []).forEach(function (row) {
      var line = make('div', 'row');
      if (row.label) {
        var label = make('div', 'row-label');
        label.appendChild(makeText('b', null, row.label));
        line.appendChild(label);
      }
      var grid = make('div', 'grid');
      grid.style.setProperty('--cols', Math.max(1, block.cols || columns.length));
      columns.forEach(function (column, index) {
        grid.appendChild(buildTile(column, (row.videos || [])[index], block.aspect, row.markAt === index));
      });
      line.appendChild(grid);
      rows.appendChild(line);
    });

    panel.appendChild(rows);
    root.appendChild(panel);

    var group = new Group(root);

    root.querySelectorAll('.frame').forEach(function (frame) {
      frame.addEventListener('click', function () {
        var video = frame.querySelector('video');
        if (video) openLightbox(video);
      });
    });

    return { root: root, group: group };
  }

  function buildCarousel(blocks) {
    var carousel = make('div', 'carousel');
    carousel.tabIndex = 0;

    var lead = blocks[0] || {};
    var shared = blocks.length > 1 && !!lead.title && blocks.every(function (block) {
      return block.title === lead.title && block.description === lead.description;
    });

    if (shared) {
      if (lead.accent) carousel.style.setProperty('--accent', lead.accent);
      carousel.appendChild(buildHead(lead));
    }

    var viewport = make('div', 'viewport');
    var track = make('div', 'track');
    viewport.appendChild(track);
    carousel.appendChild(viewport);

    var groups = [];
    var accents = [];

    blocks.forEach(function (block) {
      var slide = make('div', 'slide');
      var built = buildBlock(block, shared);
      slide.appendChild(built.root);
      track.appendChild(slide);
      built.group.selected = false;
      groups.push(built.group);
      accents.push(block.accent || '');
    });

    var nav = make('div', 'nav');
    var previous = make('button', 'arrow', PREV);
    var next = make('button', 'arrow', NEXT);
    previous.type = 'button';
    next.type = 'button';
    previous.setAttribute('aria-label', 'Previous');
    next.setAttribute('aria-label', 'Next');

    var dots = make('div', 'dots');
    var counter = makeText('span', 'counter', '');
    var center = make('div', 'nav-center');
    center.appendChild(previous);
    center.appendChild(dots);
    center.appendChild(next);
    nav.appendChild(make('span'));
    nav.appendChild(center);
    nav.appendChild(counter);
    carousel.appendChild(nav);

    var current = 0;
    var total = groups.length;

    groups.forEach(function (group, index) {
      var dot = make('button', 'dot');
      dot.type = 'button';
      dot.setAttribute('aria-label', 'Slide ' + (index + 1));
      dot.addEventListener('click', function () {
        show(index);
      });
      dots.appendChild(dot);
    });

    function show(index) {
      current = Math.max(0, Math.min(total - 1, index));
      track.style.transform = 'translateX(' + (-current * 100) + '%)';
      groups.forEach(function (group, position) {
        group.selected = position === current;
        if (Math.abs(position - current) > 1) group.unload();
        group.refresh();
      });
      if (groups[current] && groups[current].loaded) {
        [current - 1, current + 1].forEach(function (position) {
          if (groups[position]) groups[position].load();
        });
      }
      if (accents[current]) carousel.style.setProperty('--accent', accents[current]);
      dots.querySelectorAll('.dot').forEach(function (dot, position) {
        dot.classList.toggle('active', position === current);
      });
      counter.textContent = (current + 1) + ' / ' + total;
      previous.disabled = current === 0;
      next.disabled = current === total - 1;
    }

    previous.addEventListener('click', function () {
      show(current - 1);
    });
    next.addEventListener('click', function () {
      show(current + 1);
    });
    carousel.addEventListener('keydown', function (event) {
      if (event.key === 'ArrowLeft') {
        show(current - 1);
        event.preventDefault();
      }
      if (event.key === 'ArrowRight') {
        show(current + 1);
        event.preventDefault();
      }
    });

    var startX = null;
    var startY = null;
    viewport.addEventListener('pointerdown', function (event) {
      startX = event.clientX;
      startY = event.clientY;
    });
    viewport.addEventListener('pointerup', function (event) {
      if (startX === null) return;
      var dx = event.clientX - startX;
      var dy = event.clientY - startY;
      startX = null;
      if (Math.abs(dx) > 50 && Math.abs(dx) > Math.abs(dy)) show(current + (dx < 0 ? 1 : -1));
    });

    if (total <= 1) nav.remove();
    show(0);

    return { root: carousel, groups: groups };
  }

  function buildNav(sections) {
    var nav = make('nav', 'jump');
    var inner = make('div', 'jump-inner');
    sections.forEach(function (section) {
      if (!section.id || !section.heading) return;
      var link = document.createElement('a');
      link.href = '#' + section.id;
      link.textContent = section.heading;
      if (section.accent) link.style.setProperty('--accent', section.accent);
      inner.appendChild(link);
    });
    nav.appendChild(inner);
    return nav;
  }

  function buildSection(section) {
    var element = make('section', 'section');
    if (section.id) element.id = section.id;

    if (section.heading || section.lede) {
      var head = make('div', 'wrap');
      if (section.heading) head.appendChild(makeText('h2', null, section.heading));
      if (section.lede) head.appendChild(makeText('p', 'lede', section.lede));
      element.appendChild(head);
    }

    var body = make('div', 'wrap');
    var groups = [];

    if (section.layout === 'carousels' || section.layout === 'mixed') {
      (section.blocks || []).forEach(function (entry) {
        if (Array.isArray(entry)) {
          var built = buildCarousel(entry);
          body.appendChild(built.root);
          built.groups.forEach(function (group) {
            groups.push(group);
          });
        } else {
          var single = buildBlock(entry);
          body.appendChild(single.root);
          groups.push(single.group);
        }
      });
    } else if (section.layout === 'carousel') {
      var carousel = buildCarousel(section.blocks || []);
      body.appendChild(carousel.root);
      carousel.groups.forEach(function (group) {
        groups.push(group);
      });
    } else {
      (section.blocks || []).forEach(function (block) {
        var built = buildBlock(block);
        body.appendChild(built.root);
        groups.push(built.group);
      });
    }

    element.appendChild(body);
    return { root: element, groups: groups };
  }

  var lightbox = document.getElementById('lightbox');
  var lightboxInner = lightbox ? lightbox.querySelector('.lightbox-inner') : null;
  var lightboxParent = null;
  var lightboxVideo = null;

  function openLightbox(video) {
    if (!lightbox || !video) return;
    lightboxParent = video.parentNode;
    lightboxVideo = video;
    lightboxInner.appendChild(video);
    lightbox.hidden = false;
    document.body.style.overflow = 'hidden';
  }

  function closeLightbox() {
    if (!lightbox || lightbox.hidden) return;
    if (lightboxVideo && lightboxParent) {
      lightboxParent.insertBefore(lightboxVideo, lightboxParent.firstChild);
    }
    lightboxVideo = null;
    lightboxParent = null;
    lightbox.hidden = true;
    document.body.style.overflow = '';
  }

  if (lightbox) {
    lightbox.addEventListener('click', closeLightbox);
    document.getElementById('lightbox-close').addEventListener('click', closeLightbox);
    document.addEventListener('keydown', function (event) {
      if (event.key === 'Escape') closeLightbox();
    });
  }

  var allGroups = [];

  function build() {
    var data = window.PAGE_DATA || {};
    var host = document.getElementById('sections');

    if (host && (data.sections || []).length > 1) {
      host.parentNode.insertBefore(buildNav(data.sections), host);
    }

    if (host) {
      (data.sections || []).forEach(function (section) {
        var built = buildSection(section);
        host.appendChild(built.root);
        built.groups.forEach(function (group) {
          allGroups.push(group);
        });
      });
    }

    var playWatcher = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        var group = entry.target.owner;
        if (!group) return;
        group.visible = entry.isIntersecting;
        group.refresh();
      });
    }, { threshold: 0.12 });

    var loadWatcher = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        var group = entry.target.owner;
        if (entry.isIntersecting && group && group.selected) group.load();
      });
    }, { rootMargin: '400px 0px' });

    allGroups.forEach(function (group) {
      group.root.owner = group;
      playWatcher.observe(group.root);
      loadWatcher.observe(group.root);
    });

    document.addEventListener('visibilitychange', function () {
      if (document.visibilityState !== 'visible') return;
      allGroups.forEach(function (group) {
        if (group.running) group.restart();
      });
    });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', build);
  } else {
    build();
  }
})();
