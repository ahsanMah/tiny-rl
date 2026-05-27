// app.jsx — Mount design canvas
// Round 02 (top): Triptych density study  — A1 (current) + A2 (denser)
// Round 01 (below): original direction discovery — kept for reference

const { DesignCanvas, DCSection, DCArtboard, DCPostIt } = window;
const { OptionA, OptionA2, OptionA3, OptionB, OptionC, OptionD } = window;

const W = 1320;
const GAP = 56;
const PAD_X = 60;

function App() {
  return (
    <DesignCanvas>

      {/* ─── ROUND 03 · ckpt → rollout → scrub model ───────────── */}
      <DCSection id="r3-ckpt" title="Round 03 · ckpt → rollout → scrub"
                 subtitle="Conceptually corrected — eval rollouts live at checkpoints"
                 gap={GAP}>

        <DCArtboard id="a3" label="A3 · Checkpoint-aware triptych (v3 — final lo-fi)"
                    width={W} height={860}>
          <OptionA3 />
        </DCArtboard>

        <DCPostIt top={920} left={PAD_X} width={360} rotate={-1.4}>
          <strong>v3 — final lo-fi pass</strong><br/><br/>
          Primary chart split in two, both frame-level, both tied to the
          video's frame scrubber via a shared playhead.<br/><br/>
          · <strong>LEFT — cumulative_return</strong> (fixed). Shows
          the focal run's accumulated reward through the episode, with
          pinned runs' best rollouts overlaid as ghost lines. Different
          episode lengths are fine: each line just ends where its
          episode ended (v35 at 1,420f · v37 at 1,820f).<br/>
          · <strong>RIGHT — metric chart</strong> with a dropdown,
          defaulting to V̂(s_t). Other options: step_reward, action_logp,
          advantage, td_error, entropy.<br/><br/>
          <em>Dropdown over toggle:</em> 5-6 metrics is too many for a
          segmented control — it'd overflow or hide options anyway.
        </DCPostIt>

        <DCPostIt top={920} left={PAD_X + 400} width={320} rotate={1.5}>
          <strong>Checkpoint nav</strong><br/><br/>
          Mini-sparkbar (one bar per ckpt, height = eval-mean, current
          highlighted) — both indicator and scrubber. Click any bar to
          jump.<br/><br/>
          <strong>Two playheads, two semantics:</strong><br/>
          · Frame playhead (orange, solid) = current video frame, syncs
          across both upper charts.<br/>
          · Ckpt playhead (orange, dashed) = active ckpt step, syncs
          across the three secondary loss charts.<br/><br/>
          Greenlight for hi-fi when you are.
        </DCPostIt>
      </DCSection>

      {/* ─── ROUND 02 · Density study (reference) ───────────────── */}
      <DCSection id="r2-triptych" title="Round 02 · Triptych density study (reference)"
                 subtitle="Pre-checkpoint-model — kept for the density comparison"
                 gap={GAP}>

        <DCArtboard id="a1" label="A1 · Triptych (current density)"
                    width={W} height={820}>
          <OptionA />
        </DCArtboard>

        <DCArtboard id="a2" label="A2 · Triptych (denser)"
                    width={W} height={820}>
          <OptionA2 />
        </DCArtboard>
      </DCSection>

      {/* ─── ROUND 01 · Direction discovery (reference) ─────────── */}
      <DCSection id="r1-discovery" title="Round 01 · Direction discovery (reference)"
                 subtitle="Kept so we can revisit ideas to remix in"
                 gap={GAP}>

        <DCArtboard id="r1-b" label="B · Notebook Scroll"
                    width={W} height={1720}>
          <OptionB />
        </DCArtboard>
        <DCArtboard id="r1-c" label="C · Bento Canvas"
                    width={W} height={944}>
          <OptionC />
        </DCArtboard>
        <DCArtboard id="r1-d" label="D · Pinned Columns (Compare)"
                    width={W} height={900}>
          <OptionD />
        </DCArtboard>
      </DCSection>

    </DesignCanvas>
  );
}

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(<App />);
