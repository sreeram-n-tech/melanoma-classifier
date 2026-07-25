/*
  Presentation + preprocessing wiring. Talks to the same /predict and /compare
  endpoints; the honest inference path (webapp/inference.py) is untouched —
  preprocessing runs server-side, before inference, via
  webapp/preprocessing_bridge.py. /compare calls inference.infer_pil once per
  registered model (no ensembling, no new math) and never averages the two
  models' probabilities into a single score.
*/
(function () {
  const fileEl = document.getElementById('file');
  const modelSelectEl = document.getElementById('model');
  const compareModeEl = document.getElementById('compare-mode');
  const runEl = document.getElementById('run');
  const errEl = document.getElementById('err');
  const resultEl = document.getElementById('result');
  const compareResultEl = document.getElementById('compare-result');
  const emptyEl = document.getElementById('empty-state');
  const readoutEl = document.getElementById('readout');
  const markerEl = document.getElementById('readout-marker');
  const markerLabelEl = document.getElementById('marker-label');
  const gateLabelEl = document.getElementById('gate-label');
  const beforeAfterEl = document.getElementById('before-after');
  const baBeforeImg = document.getElementById('ba-before');
  const baAfterImg = document.getElementById('ba-after');
  const provisionalBadge = document.getElementById('provisional-badge');
  const offDistCaveat = document.getElementById('off-dist-caveat');
  const stepsReportEl = document.getElementById('steps-report');

  const cmpBeforeAfterEl = document.getElementById('cmp-before-after');
  const cmpBaBeforeImg = document.getElementById('cmp-ba-before');
  const cmpBaAfterImg = document.getElementById('cmp-ba-after');
  const cmpOffDistCaveat = document.getElementById('cmp-off-dist-caveat');
  const cmpVerdictEl = document.getElementById('compare-verdict');
  const cmpStepsReportEl = document.getElementById('cmp-steps-report');

  const colA = {
    name: document.getElementById('cmp-name-a'),
    prob: document.getElementById('cmp-prob-a'),
    badge: document.getElementById('cmp-badge-a'),
    flag: document.getElementById('cmp-flag-a'),
    readout: document.getElementById('cmp-readout-a'),
    marker: document.getElementById('cmp-marker-a'),
    markerLabel: document.getElementById('cmp-marker-label-a'),
    gateLabel: document.getElementById('cmp-gate-a'),
    thresholdNote: document.getElementById('cmp-threshold-note-a'),
    stats: document.getElementById('cmp-stats-a'),
  };
  const colB = {
    name: document.getElementById('cmp-name-b'),
    prob: document.getElementById('cmp-prob-b'),
    badge: document.getElementById('cmp-badge-b'),
    flag: document.getElementById('cmp-flag-b'),
    readout: document.getElementById('cmp-readout-b'),
    marker: document.getElementById('cmp-marker-b'),
    markerLabel: document.getElementById('cmp-marker-label-b'),
    gateLabel: document.getElementById('cmp-gate-b'),
    thresholdNote: document.getElementById('cmp-threshold-note-b'),
    stats: document.getElementById('cmp-stats-b'),
  };

  let lastObjectUrl = null;

  fileEl.addEventListener('change', () => {
    runEl.disabled = !fileEl.files.length;
    errEl.textContent = '';
  });

  compareModeEl.addEventListener('change', () => {
    modelSelectEl.disabled = compareModeEl.checked;
  });

  runEl.addEventListener('click', async () => {
    if (!fileEl.files.length) return;
    errEl.textContent = '';
    runEl.disabled = true;
    runEl.textContent = 'Reading…';
    const compareMode = compareModeEl.checked;
    try {
      const checked = Array.from(document.querySelectorAll('.preproc-toggle:checked'))
        .map((el) => el.value);

      const fd = new FormData();
      fd.append('file', fileEl.files[0]);
      checked.forEach((key) => fd.append('preprocess', key));

      let endpoint = '/predict';
      if (compareMode) {
        endpoint = '/compare';
      } else {
        fd.append('model_key', modelSelectEl.value);
      }

      const res = await fetch(endpoint, { method: 'POST', body: fd });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.detail || 'Something went wrong reading that image.');
      }
      const data = await res.json();
      if (compareMode) {
        renderCompare(data, fileEl.files[0]);
      } else {
        renderSingle(data, fileEl.files[0]);
      }
    } catch (e) {
      errEl.textContent = e.message;
      resultEl.hidden = true;
      compareResultEl.hidden = true;
      emptyEl.hidden = false;
    } finally {
      runEl.disabled = false;
      runEl.textContent = 'Run reading';
    }
  });

  // Fills one reading's fields (prob, flag, readout, threshold note, stats,
  // provisional badge) into a set of DOM elements. Used for both the single
  // reading and each column of the side-by-side compare view, so the two
  // modes can never drift out of sync with each other.
  function fillReading(els, r) {
    els.prob.textContent = r.percent + '%  P(melanoma)';

    if (r.above_threshold) {
      els.flag.className = 'flag refer';
      els.flag.textContent = 'Above the model’s referral threshold — a clinician should review any real concern';
    } else {
      els.flag.className = 'flag below';
      els.flag.textContent = 'Below the model’s referral threshold';
    }

    const thresholdPct = r.threshold * 100;
    const probPct = r.percent;
    els.readout.style.setProperty('--threshold-pct', thresholdPct);
    els.marker.style.left = probPct + '%';
    els.markerLabel.textContent = probPct.toFixed(1) + '%';
    els.gateLabel.textContent = r.threshold.toFixed(3) + ' threshold';
    els.readout.setAttribute(
      'aria-label',
      'Calibration readout: ' + probPct.toFixed(1) + ' percent probability of melanoma, ' +
      (r.above_threshold ? 'above' : 'below') + ' the ' + thresholdPct.toFixed(1) +
      ' percent referral threshold.'
    );

    els.thresholdNote.innerHTML =
      'Decision uses the frozen referral threshold <span class="mono">' +
      r.threshold.toFixed(3) + '</span> (tuned for 95% sensitivity on validation — not a naïve 0.50). ' +
      'Temperature-calibrated with <span class="mono">T=' + r.temperature.toFixed(3) + '</span>' +
      (r.tta ? ', 4-flip TTA.' : '.');

    els.stats.innerHTML =
      r.model_name + ' — known held-out performance: ' +
      'AUC <span class="mono stat-value">' + r.stats.auc.toFixed(3) + '</span> · ' +
      'Sensitivity <span class="mono stat-value">' + r.stats.sensitivity.toFixed(2) + '</span> · ' +
      'Specificity <span class="mono stat-value">' + r.stats.specificity.toFixed(2) + '</span>.';

    els.badge.hidden = !r.preprocessing_applied;
    els.readout.classList.toggle('provisional', !!r.preprocessing_applied);
  }

  function renderStepsReport(container, steps) {
    if (steps && steps.length) {
      container.innerHTML = steps.map((s) => {
        const warn = s.warning ? ' <span class="warn">' + s.warning + '</span>' : '';
        return '<div class="step"><span class="step-label">' + s.label + ':</span> ' + s.caption + warn + '</div>';
      }).join('');
    } else {
      container.innerHTML = '';
    }
  }

  function renderSingle(r, uploadedFile) {
    fillReading({
      prob: document.getElementById('prob'),
      badge: provisionalBadge,
      flag: document.getElementById('flag'),
      readout: readoutEl,
      marker: markerEl,
      markerLabel: markerLabelEl,
      gateLabel: gateLabelEl,
      thresholdNote: document.getElementById('threshold-note'),
      stats: document.getElementById('stats'),
    }, r);

    document.getElementById('calib-note').textContent =
      'This is a research model’s estimate, not a diagnosis. Shown with its real error rates so the number is read in context.';

    offDistCaveat.hidden = !r.preprocessing_applied;

    if (r.preprocessing_applied && r.processed_image_b64) {
      if (lastObjectUrl) URL.revokeObjectURL(lastObjectUrl);
      lastObjectUrl = URL.createObjectURL(uploadedFile);
      baBeforeImg.src = lastObjectUrl;
      baAfterImg.src = 'data:image/png;base64,' + r.processed_image_b64;
      beforeAfterEl.hidden = false;
    } else {
      beforeAfterEl.hidden = true;
    }

    renderStepsReport(stepsReportEl, r.steps);

    emptyEl.hidden = true;
    compareResultEl.hidden = true;
    resultEl.hidden = false;
  }

  function renderCompare(data, uploadedFile) {
    const [ra, rb] = data.results;
    colA.name.textContent = ra.model_name;
    colB.name.textContent = rb.model_name;
    fillReading(colA, ra);
    fillReading(colB, rb);

    if (data.disagree) {
      cmpVerdictEl.className = 'compare-verdict disagree';
      cmpVerdictEl.textContent =
        'The two models disagree on whether this crosses their referral threshold — that’s ' +
        'real information, not resolved here. Independently-trained, independently-calibrated ' +
        'models will not always agree.';
    } else {
      cmpVerdictEl.className = 'compare-verdict agree';
      const dir = ra.above_threshold ? 'above' : 'below';
      cmpVerdictEl.textContent = 'Both models agree: ' + dir + ' their own referral threshold.';
    }

    cmpOffDistCaveat.hidden = !data.preprocessing_applied;

    if (data.preprocessing_applied && data.processed_image_b64) {
      if (lastObjectUrl) URL.revokeObjectURL(lastObjectUrl);
      lastObjectUrl = URL.createObjectURL(uploadedFile);
      cmpBaBeforeImg.src = lastObjectUrl;
      cmpBaAfterImg.src = 'data:image/png;base64,' + data.processed_image_b64;
      cmpBeforeAfterEl.hidden = false;
    } else {
      cmpBeforeAfterEl.hidden = true;
    }

    renderStepsReport(cmpStepsReportEl, data.steps);

    emptyEl.hidden = true;
    resultEl.hidden = true;
    compareResultEl.hidden = false;
  }
})();
