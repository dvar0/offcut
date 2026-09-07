// UI counterpart of Comfy's Flux/simple schedule and offcut_cli.split_sigma_schedules.
// This is arithmetic only: a step preview must not need a loaded engine or a GPU request.
const f32 = Math.fround;

function simpleSchedule(shift, steps) {
  const exp = f32(Math.exp(shift));
  return Array.from({ length: steps }, (_, index) => {
    const time = f32((10000 - Math.floor(index * (10000 / steps))) / 10000);
    return f32(exp / f32(exp + f32(f32(1 / time) - 1)));
  });
}

// Python rounds exact halves to even; Math.round does not. Historical percentages must select
// the same raw boundary here as they do in the engine, including a hand-typed half step.
export function rawStepCount(portion, density) {
  const value = density * portion / 100;
  const floor = Math.floor(value);
  const rounded = value - floor === 0.5 ? floor + floor % 2 : Math.round(value);
  return Math.max(1, Math.min(density - 1, rounded));
}

export function hybridStepPlan(width, height, portion = 8, rawDensity = 52, turboDensity = 12) {
  if (![width, height, portion, rawDensity, turboDensity].every(Number.isFinite)
      || width <= 0 || height <= 0 || portion <= 0 || portion >= 100
      || !Number.isInteger(rawDensity) || rawDensity < 2 || rawDensity > 100
      || !Number.isInteger(turboDensity) || turboDensity < 2 || turboDensity > 100) return null;
  const tokens = Math.ceil(width / 16) * Math.ceil(height / 16);
  const shift = 0.5 + (tokens - 256) * (1.15 - 0.5) / (6400 - 256);
  const raw = rawStepCount(portion, rawDensity);
  const boundary = simpleSchedule(shift, rawDensity)[raw];
  const turboSchedule = simpleSchedule(1.15, turboDensity);
  let start = 1;
  let distance = Infinity;
  for (let index = 1; index < turboDensity; index += 1) {
    const candidate = Math.abs(turboSchedule[index] - boundary);
    if (turboSchedule[index - 1] > boundary && candidate < distance) {
      start = index;
      distance = candidate;
    }
  }
  return { raw, turbo: turboDensity - start };
}

export function stepPlanText(plan) {
  return `${plan.raw} Raw ${plan.raw === 1 ? "step" : "steps"} → ${plan.turbo} Turbo ${plan.turbo === 1 ? "step" : "steps"}`;
}

// Both the Create panel and cover recipe use the same stepper. The hidden percentage remains
// the saved recipe, and is preserved verbatim until the user edits the count or raw density.
export function bindHybridControls(ids) {
  const fields = Object.fromEntries(Object.entries(ids).map(([key, id]) => [key, document.getElementById(id)]));
  const density = () => fields.rawDensity.value === "" ? 52 : Number(fields.rawDensity.value);
  const portion = () => fields.portion.value === "" ? 8 : Number(fields.portion.value);

  function refresh() {
    const rawDensity = density();
    const plan = hybridStepPlan(Number(fields.width.value), Number(fields.height.value), portion(), rawDensity,
      fields.turboDensity.value === "" ? 12 : Number(fields.turboDensity.value));
    if (plan) fields.count.value = plan.raw;
    fields.count.max = Math.max(1, rawDensity - 1);
    fields.readout.textContent = plan ? stepPlanText(plan) : "Enter valid sampling settings to see the step counts.";
    fields.minus.disabled = !plan || plan.raw <= 1;
    fields.plus.disabled = !plan || plan.raw >= rawDensity - 1;
  }

  function commitCount() {
    const count = Number(fields.count.value);
    const rawDensity = density();
    if (!fields.count.value || !Number.isInteger(count) || !Number.isInteger(rawDensity)
        || rawDensity < 2 || rawDensity > 100 || count < 1 || count >= rawDensity) {
      fields.readout.textContent = "Choose a Raw step count within the available range.";
      return;
    }
    fields.portion.value = count / rawDensity * 100;
    refresh();
  }

  fields.count.addEventListener("input", commitCount);
  fields.rawDensity.addEventListener("input", () => {
    const rawDensity = density();
    if (Number.isInteger(rawDensity) && rawDensity >= 2 && rawDensity <= 100) {
      fields.count.value = Math.max(1, Math.min(rawDensity - 1, Number(fields.count.value) || 1));
      commitCount();
    } else refresh();
  });
  for (const field of [fields.turboDensity, fields.width, fields.height]) field.addEventListener("input", refresh);
  for (const [button, delta] of [[fields.minus, -1], [fields.plus, 1]]) {
    button.addEventListener("click", () => {
      fields.count.value = Math.max(1, Math.min(density() - 1, Number(fields.count.value) + delta));
      fields.count.dispatchEvent(new Event("input", { bubbles: true }));
    });
  }
  return { refresh };
}
