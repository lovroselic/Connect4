/*jshint browser: true */
/*jshint esversion: 11 */
/*jshint -W097 */
/*global INI */

"use strict";

/*
  Connect4LookaheadJS
  - Bitboard alpha-beta (negamax) with TT + killer/history ordering
  - Tactical guards:
      DOUBLE_THREAT_GUARD: avoid moves that give opponent >=2 win-in-1 replies
      FORK_REPLY_GUARD: avoid moves where opponent can reply to create >=2 win-in-1 threats
  - Heuristic: floating penalties (per empty), center, parity, tempo, immediate wins/forks
  - API mirrors Kaggle version Python class
*/

const C4LA_DEFAULTS = {
    WIN2_CHECK: true,

    DOUBLE_THREAT_GUARD: true,
    FORK_REPLY_GUARD: true,

    C4_WIN: 100000.0,
    C4_IMMEDIATE_W: 100000.0,
    C4_FORK_W: 100000.0,

    C4_DEFENSIVE: 1.55,
    C4_FLOATING_NEAR: 0.25,
    C4_FLOATING_FAR: 0.125,
    C4_CENTER_BONUS: 3.0,
    C4_PARITY_BONUS: 0.75,

    C4_VERT_MUL: 0.80,
    C4_VERT_3_READY_BONUS: 0.0,
    C4_TEMPO_W: 75,
    C4_PARITY_MOVE_W: 0.5,
    C4_PARITY_UNLOCK_W: 0.25,
    C4_THREATSPACE_W: 9.0,

    C4_DEFAULT_WEIGHTS_ITEMS: [
        [2, 10.0],
        [3, 1000.0],
        [4, 100000.0],
    ],

    C4_SOFT_MATE_MULT: 100.0,
    C4_MATE_SCORE_MULT: 1000.0,

    TT_SIZE_POW2: 16,          // 1<<16 entries
    TT_MAX_PROBES: 32,
};

class Connect4LookaheadJS {
    // board constants
    static ROWS = 6;
    static COLS = 7;
    static K = 4;
    static STRIDE = Connect4LookaheadJS.ROWS + 1; // 7
    static CENTER_COL = 3;

    static _CENTER_ORDER = [3, 4, 2, 5, 1, 6, 0];

    // precomp (static, BigInt)
    static _PRECOMP_DONE = false;
    static COL_MASK = new Array(Connect4LookaheadJS.COLS).fill(0n);
    static TOP_MASK = new Array(Connect4LookaheadJS.COLS).fill(0n);
    static BOTTOM_MASK = new Array(Connect4LookaheadJS.COLS).fill(0n);
    static FULL_MASK = 0n;
    static CENTER_MASK = 0n;

    static ODD_MASK = 0n;
    static EVEN_MASK = 0n;

    // window data
    static WIN_MASKS = [];     // BigInt[]
    static WIN_KIND = [];      // number[] (0 horiz,1 vert,2 diagUR,3 diagUL)
    static WIN_B = [];         // flattened BigInt[W*4] bit per cell
    static WIN_C = [];         // flattened number[W*4] col per cell
    static WIN_R = [];         // flattened number[W*4] row per cell (bottom-based)

    // instance-friendly accessors for static constants
    get ROWS() { return Connect4LookaheadJS.ROWS; }
    get COLS() { return Connect4LookaheadJS.COLS; }
    get STRIDE() { return Connect4LookaheadJS.STRIDE; }
    get K() { return Connect4LookaheadJS.K; }


    // opening
    OPENING_BOOK = true;
    OPENING_RANDOM = true;

    // weights + knobs (instance)
    immediate_w = C4LA_DEFAULTS.C4_IMMEDIATE_W;
    fork_w = C4LA_DEFAULTS.C4_FORK_W;
    DEFENSIVE = C4LA_DEFAULTS.C4_DEFENSIVE;
    FLOATING_NEAR = C4LA_DEFAULTS.C4_FLOATING_NEAR;
    FLOATING_FAR = C4LA_DEFAULTS.C4_FLOATING_FAR;
    CENTER_BONUS = C4LA_DEFAULTS.C4_CENTER_BONUS;
    PARITY_BONUS = C4LA_DEFAULTS.C4_PARITY_BONUS;

    VERT_MUL = C4LA_DEFAULTS.C4_VERT_MUL;
    VERT_3_READY_BONUS = C4LA_DEFAULTS.C4_VERT_3_READY_BONUS;
    TEMPO_W = C4LA_DEFAULTS.C4_TEMPO_W;
    PARITY_MOVE_W = C4LA_DEFAULTS.C4_PARITY_MOVE_W;
    PARITY_UNLOCK_W = C4LA_DEFAULTS.C4_PARITY_UNLOCK_W;
    THREATSPACE_W = C4LA_DEFAULTS.C4_THREATSPACE_W;

    DOUBLE_THREAT_GUARD = C4LA_DEFAULTS.DOUBLE_THREAT_GUARD;
    FORK_REPLY_GUARD = C4LA_DEFAULTS.FORK_REPLY_GUARD;

    WIN2_CHECK = C4LA_DEFAULTS.WIN2_CHECK;

    constructor(weights = null, opts = null) {
        if (!Connect4LookaheadJS._PRECOMP_DONE) {
            Connect4LookaheadJS._build_precomp();
        }

        // weights map: {2:...,3:...,4:...}
        this.weights = new Map(C4LA_DEFAULTS.C4_DEFAULT_WEIGHTS_ITEMS);
        if (weights) {
            for (const k of Object.keys(weights)) {
                this.weights.set(Number(k), Number(weights[k]));
            }
        }

        // derived mate scales
        this.SOFT_MATE = C4LA_DEFAULTS.C4_SOFT_MATE_MULT * this.weights.get(4);
        this.MATE_SCORE = C4LA_DEFAULTS.C4_MATE_SCORE_MULT * this.weights.get(4);

        // optional overrides
        if (opts && typeof opts === "object") {
            for (const [k, v] of Object.entries(opts)) {
                if (k in this) this[k] = v;
            }
        }
    }

    // ---------------- Public API ----------------

    get_heuristic(board, player) {
        const bb = this._parse_board_bitboards(board);
        const meMark = this._p(player); // 1 or -1
        const me = (meMark === 1) ? bb.p1 : bb.p2;

        const warr = this._makeWArr();
        const role = this._role_first_from_center(bb.mask, bb.p1, bb.p2);
        const root_pos_is_first = (role.parityEnabled && meMark === role.roleFirstMark) ? 1 : 0;

        return this._evaluate(
            me, bb.mask,
            warr,
            role.parityEnabled ? 1 : 0,
            root_pos_is_first,
            0
        );
    }

    is_terminal(board) {
        const bb = this._parse_board_bitboards(board);
        return (
            this._has_won(bb.p1) ||
            this._has_won(bb.p2) ||
            (bb.mask === Connect4LookaheadJS.FULL_MASK)
        );
    }

    minimax(board, depth, maximizing, player, alpha, beta) {
        const bb = this._parse_board_bitboards(board);

        const rootPov = this._p(player);              // 1 or -1
        const toMove = maximizing ? rootPov : -rootPov;
        const pos = (toMove === 1) ? bb.p1 : bb.p2;   // side-to-move bitboard

        const warr = this._makeWArr();

        const role = this._role_first_from_center(bb.mask, bb.p1, bb.p2);
        const root_pos_is_first = (role.parityEnabled && toMove === role.roleFirstMark) ? 1 : 0;

        const ctx = this._makeSearchContext(warr);
        const res = this._negamax(
            pos, bb.mask,
            depth | 0,
            Number(alpha), Number(beta),
            0,
            root_pos_is_first,
            role.parityEnabled ? 1 : 0,
            (this.DOUBLE_THREAT_GUARD ? 1 : 0),
            (this.FORK_REPLY_GUARD ? 1 : 0),
            ctx
        );

        const val = res.val;
        // if caller asked from non-to-move POV, flip to mimic Python wrapper
        return (toMove === rootPov) ? val : -val;
    }

    n_step_lookahead(board, player, depth = 3) {
        const bb = this._parse_board_bitboards(board);
        const stones = bb.stones;
        const meMark = this._p(player); // 1 or -1

        // opening book (kept compatible with KAggle Python version)
        if (this.OPENING_BOOK) {
            if (stones === 0 && meMark === 1) return Connect4LookaheadJS.CENTER_COL;

            if (stones === 1 && meMark === -1) {
                const b_d1 = 1n << BigInt(Connect4LookaheadJS.CENTER_COL * Connect4LookaheadJS.STRIDE + 0);
                if ((bb.mask & b_d1) !== 0n) {
                    const choices = [2, 4];
                    return this.OPENING_RANDOM ? choices[(Math.random() < 0.5) ? 0 : 1] : choices[0];
                }
            }

            if (stones === 2 && meMark === 1) {
                const s = Connect4LookaheadJS.STRIDE;
                const cc = Connect4LookaheadJS.CENTER_COL;
                const b_d1 = 1n << BigInt(cc * s + 0);
                if ((bb.mask & b_d1) !== 0n) {
                    const b_d2 = 1n << BigInt(cc * s + 1);
                    const b_a1 = 1n << BigInt(0 * s + 0);
                    const b_b1 = 1n << BigInt(1 * s + 0);
                    const b_c1 = 1n << BigInt(2 * s + 0);
                    const b_e1 = 1n << BigInt(4 * s + 0);
                    const b_f1 = 1n << BigInt(5 * s + 0);
                    const b_g1 = 1n << BigInt(6 * s + 0);

                    if ((bb.mask & b_d2) !== 0n) {
                        if (this._can_play(bb.mask, cc)) return cc;
                    }
                    if ((bb.mask & b_c1) !== 0n) return 5;
                    if ((bb.mask & b_e1) !== 0n) return 1;
                    if ((bb.mask & b_a1) !== 0n) return 4;
                    if ((bb.mask & b_g1) !== 0n) return 2;
                    if ((bb.mask & b_b1) !== 0n) return 5;
                    if ((bb.mask & b_f1) !== 0n) return 1;
                    if (this._can_play(bb.mask, cc)) return cc;
                }
            }

            if (stones === 3 && meMark === -1) {
                const s = Connect4LookaheadJS.STRIDE;
                const cc = Connect4LookaheadJS.CENTER_COL;
                const b_d1 = 1n << BigInt(cc * s + 0);
                const b_d2 = 1n << BigInt(cc * s + 1);
                const b_d3 = 1n << BigInt(cc * s + 2);
                if ((bb.mask & b_d1) && (bb.mask & b_d2) && (bb.mask & b_d3)) {
                    if (this._can_play(bb.mask, cc)) return cc;
                }
            }
        }

        const me = (meMark === 1) ? bb.p1 : bb.p2;
        const warr = this._makeWArr();
        const role = this._role_first_from_center(bb.mask, bb.p1, bb.p2);
        const root_pos_is_first = (role.parityEnabled && meMark === role.roleFirstMark) ? 1 : 0;
        const ctx = this._makeSearchContext(warr);

        return this._root_select_fixed(
            me, bb.mask,
            depth | 0,
            root_pos_is_first,
            role.parityEnabled ? 1 : 0,
            (this.DOUBLE_THREAT_GUARD ? 1 : 0),
            (this.FORK_REPLY_GUARD ? 1 : 0),
            ctx
        );
    }

    has_four(board, player) {
        const bb = this._parse_board_bitboards(board);
        const me = (this._p(player) === 1) ? bb.p1 : bb.p2;
        return this._has_won(me);
    }

    check_win(board, player) {
        return this.has_four(board, player);
    }

    count_immediate_wins(board, player) {
        const bb = this._parse_board_bitboards(board);
        const me = (this._p(player) === 1) ? bb.p1 : bb.p2;

        const wins = [];
        for (const c of Connect4LookaheadJS._CENTER_ORDER) {
            if ((bb.mask & Connect4LookaheadJS.TOP_MASK[c]) !== 0n) continue;
            if (this._is_winning_move(me, bb.mask, c)) wins.push(c);
        }
        return wins;
    }

    compute_fork_signals(board_before, board_after, mover) {
        const moverMark = this._p(mover);
        const opp = -moverMark;
        const my_after = this.count_immediate_wins(board_after, moverMark).length;
        const opp_before = this.count_immediate_wins(board_before, opp).length;
        const opp_after = this.count_immediate_wins(board_after, opp).length;
        return { my_after, opp_before, opp_after };
    }

    count_pure(board, player, n) {
        const bb = this._parse_board_bitboards(board);
        const me = (this._p(player) === 1) ? bb.p1 : bb.p2;
        const opp = (this._p(player) === 1) ? bb.p2 : bb.p1;

        let cnt = 0;
        for (let i = 0; i < Connect4LookaheadJS.WIN_MASKS.length; i++) {
            const wmask = Connect4LookaheadJS.WIN_MASKS[i];
            const mp = wmask & me;
            const mo = wmask & opp;
            if (mo === 0n && this._popcount(mp) === (n | 0)) cnt++;
        }
        return cnt;
    }

    count_pure_block_delta(before_board, after_board, player, n) {
        const before = this.count_pure(before_board, player, n);
        const after = this.count_pure(after_board, player, n);
        return Math.max(0, before - after);
    }

    n_step_action_scores(board, player, depth = 1) {
        const bb = this._parse_board_bitboards(board);
        const meMark = this._p(player);
        const me = (meMark === 1) ? bb.p1 : bb.p2;

        const scores = new Float64Array(Connect4LookaheadJS.COLS);
        for (let i = 0; i < scores.length; i++) scores[i] = -Infinity;

        const warr = this._makeWArr();
        const role = this._role_first_from_center(bb.mask, bb.p1, bb.p2);
        const root_pos_is_first = (role.parityEnabled && meMark === role.roleFirstMark) ? 1 : 0;

        for (const c of Connect4LookaheadJS._CENTER_ORDER) {
            if ((bb.mask & Connect4LookaheadJS.TOP_MASK[c]) !== 0n) continue;

            if (this._is_winning_move(me, bb.mask, c)) {
                scores[c] = this.MATE_SCORE;
                continue;
            }

            const mv = this._play_bit(bb.mask, c);
            const nm = bb.mask | mv;
            const next_pos = nm ^ (me | mv); // opponent-to-move after my move

            const ctx = this._makeSearchContext(warr);

            const res = this._negamax(
                next_pos, nm,
                (depth - 1) | 0,
                -this.MATE_SCORE, this.MATE_SCORE,
                1,
                root_pos_is_first,
                role.parityEnabled ? 1 : 0,
                (this.DOUBLE_THREAT_GUARD ? 1 : 0),
                (this.FORK_REPLY_GUARD ? 1 : 0),
                ctx
            );

            scores[c] = -res.val;
        }

        return scores;
    }

    policy_scores_delta(board, player, depth = 1) {
        const base = this.get_heuristic(board, player);
        const sc = this.n_step_action_scores(board, player, depth);
        const out = new Float64Array(sc.length);
        for (let i = 0; i < sc.length; i++) out[i] = sc[i] - base;
        return out;
    }

    legal_actions(board = null, mask = null) {
        let m = mask;
        if (m == null) {
            if (board == null) throw new Error("legal_actions() requires either board or mask");
            m = this._parse_board_bitboards(board).mask;
        }
        const out = [];
        for (let c = 0; c < Connect4LookaheadJS.COLS; c++) {
            if ((m & Connect4LookaheadJS.TOP_MASK[c]) === 0n) out.push(c);
        }
        return out;
    }

    random_action(board) {
        const legal = this.legal_actions(board);
        if (!legal.length) return -1;
        return legal[(Math.random() * legal.length) | 0] | 0;
    }

    leftmost_action(board) {
        const legal = this.legal_actions(board);
        if (!legal.length) return -1;
        let best = legal[0];
        for (let i = 1; i < legal.length; i++) best = Math.min(best, legal[i]);
        return best | 0;
    }

    baseline_action(board, kind = "random") {
        const k = String(kind || "").toLowerCase();
        if (k === "random" || k === "rnd") return this.random_action(board);
        if (k === "leftmost" || k === "left" || k === "lm") return this.leftmost_action(board);
        if (k === "center" || k === "centre") {
            const bb = this._parse_board_bitboards(board);
            for (const c of Connect4LookaheadJS._CENTER_ORDER) {
                if ((bb.mask & Connect4LookaheadJS.TOP_MASK[c]) === 0n) return c | 0;
            }
            return 0;
        }
        throw new Error("Unknown baseline kind: " + kind);
    }

    baseline_policy_probs(board, kind = "random") {
        const probs = new Float32Array(Connect4LookaheadJS.COLS);
        const bb = this._parse_board_bitboards(board);
        const k = String(kind || "").toLowerCase();

        if (k === "random" || k === "rnd") {
            let legal = 0;
            for (let c = 0; c < Connect4LookaheadJS.COLS; c++) {
                if ((bb.mask & Connect4LookaheadJS.TOP_MASK[c]) === 0n) legal++;
            }
            if (legal === 0) { probs[0] = 1.0; return probs; }
            const p = 1.0 / legal;
            for (let c = 0; c < Connect4LookaheadJS.COLS; c++) {
                if ((bb.mask & Connect4LookaheadJS.TOP_MASK[c]) === 0n) probs[c] = p;
            }
            return probs;
        }

        if (k === "leftmost" || k === "left" || k === "lm") {
            for (let c = 0; c < Connect4LookaheadJS.COLS; c++) {
                if ((bb.mask & Connect4LookaheadJS.TOP_MASK[c]) === 0n) { probs[c] = 1.0; return probs; }
            }
            probs[0] = 1.0;
            return probs;
        }

        if (k === "center" || k === "centre") {
            for (const c of Connect4LookaheadJS._CENTER_ORDER) {
                if ((bb.mask & Connect4LookaheadJS.TOP_MASK[c]) === 0n) { probs[c] = 1.0; return probs; }
            }
            probs[0] = 1.0;
            return probs;
        }

        throw new Error("Unknown baseline kind: " + kind);
    }

    sample_action_from_probs(probs) {
        const p = Array.from(probs, x => Number(x));
        let s = 0;
        for (let i = 0; i < p.length; i++) s += p[i];
        if (s <= 0) return 0;

        let r = Math.random() * s;
        for (let i = 0; i < p.length; i++) {
            r -= p[i];
            if (r <= 0) return i | 0;
        }
        return (p.length - 1) | 0;
    }

    // ---------------- Internals: precompute ----------------

    static _bit_at(c, r) {
        return 1n << BigInt(c * Connect4LookaheadJS.STRIDE + r);
    }

    static _build_precomp() {
        // masks
        let full = 0n;
        for (let c = 0; c < Connect4LookaheadJS.COLS; c++) {
            let colBits = 0n;
            for (let r = 0; r < Connect4LookaheadJS.ROWS; r++) {
                colBits |= Connect4LookaheadJS._bit_at(c, r);
            }
            Connect4LookaheadJS.COL_MASK[c] = colBits;
            Connect4LookaheadJS.BOTTOM_MASK[c] = Connect4LookaheadJS._bit_at(c, 0);
            Connect4LookaheadJS.TOP_MASK[c] = Connect4LookaheadJS._bit_at(c, Connect4LookaheadJS.ROWS - 1);
            full |= colBits;
        }
        Connect4LookaheadJS.FULL_MASK = full;
        Connect4LookaheadJS.CENTER_MASK = Connect4LookaheadJS.COL_MASK[Connect4LookaheadJS.CENTER_COL];

        // parity masks (bottom-based r=0 is "odd" in your Python)
        let odd = 0n, even = 0n;
        for (let c = 0; c < Connect4LookaheadJS.COLS; c++) {
            for (let r = 0; r < Connect4LookaheadJS.ROWS; r++) {
                const b = Connect4LookaheadJS._bit_at(c, r);
                if ((r & 1) === 0) odd |= b;
                else even |= b;
            }
        }
        Connect4LookaheadJS.ODD_MASK = odd;
        Connect4LookaheadJS.EVEN_MASK = even;

        // windows (69)
        const addWindow = (cells, kind) => {
            let mask = 0n;
            for (let i = 0; i < 4; i++) {
                const cc = cells[i][0];
                const rr = cells[i][1];
                const b = Connect4LookaheadJS._bit_at(cc, rr);
                mask |= b;
                Connect4LookaheadJS.WIN_B.push(b);
                Connect4LookaheadJS.WIN_C.push(cc);
                Connect4LookaheadJS.WIN_R.push(rr);
            }
            Connect4LookaheadJS.WIN_MASKS.push(mask);
            Connect4LookaheadJS.WIN_KIND.push(kind);
        };

        // horiz
        for (let r = 0; r < Connect4LookaheadJS.ROWS; r++) {
            for (let c = 0; c <= Connect4LookaheadJS.COLS - 4; c++) {
                addWindow([[c, r], [c + 1, r], [c + 2, r], [c + 3, r]], 0);
            }
        }
        // vert
        for (let c = 0; c < Connect4LookaheadJS.COLS; c++) {
            for (let r = 0; r <= Connect4LookaheadJS.ROWS - 4; r++) {
                addWindow([[c, r], [c, r + 1], [c, r + 2], [c, r + 3]], 1);
            }
        }
        // diag up-right
        for (let c = 0; c <= Connect4LookaheadJS.COLS - 4; c++) {
            for (let r = 0; r <= Connect4LookaheadJS.ROWS - 4; r++) {
                addWindow([[c, r], [c + 1, r + 1], [c + 2, r + 2], [c + 3, r + 3]], 2);
            }
        }
        // diag up-left
        for (let c = 3; c < Connect4LookaheadJS.COLS; c++) {
            for (let r = 0; r <= Connect4LookaheadJS.ROWS - 4; r++) {
                addWindow([[c, r], [c - 1, r + 1], [c - 2, r + 2], [c - 3, r + 3]], 3);
            }
        }

        Connect4LookaheadJS._PRECOMP_DONE = true;
    }

    // ---------------- Internals: parsing ----------------

    _p(p) {
        // Accept 1/2 or 1/-1
        if (p === 2) return -1;
        if (p === -1) return -1;
        return 1;
    }


    _parse_board_bitboards(board) {
        const ROWS = this.ROWS;
        const COLS = this.COLS;
        const STRIDE = this.STRIDE;

        const flat = this._boardToFlatBottomFirst(board); // length = 42, y=0 bottom row

        let p1 = 0n, p2 = 0n, mask = 0n;

        for (let y = 0; y < ROWS; y++) {
            const rowOff = y * COLS;
            for (let x = 0; x < COLS; x++) {
                const v = flat[rowOff + x] | 0;
                if (v === 0) continue;

                const b = 1n << BigInt(x * STRIDE + y);
                mask |= b;
                if (v === 1) p1 |= b;
                else p2 |= b; // assumes 2
            }
        }

        const stones = this._popcount(mask);
        return { p1, p2, mask, stones };
    }

    _boardToFlatBottomFirst(board) {
        const ROWS = this.ROWS, COLS = this.COLS;
        const N = ROWS * COLS;            // 42
        const N_PAD = (ROWS + 1) * COLS;  // 49 (sentinel row)

        // C4Grid / GA: board.map is the flat buffer
        if (board && board.map && (Array.isArray(board.map) || ArrayBuffer.isView(board.map))) {
            const a = board.map;

            if (a.length === N) return a;

            // your engine stores 7 rows (6+sentinel). We ignore the last row (y=6).
            if (a.length === N_PAD) {
                // Array -> slice, TypedArray -> subarray
                return (typeof a.subarray === "function") ? a.subarray(0, N) : a.slice(0, N);
            }

            throw new Error(`Unsupported GA.map length ${a.length}, expected ${N} or ${N_PAD}.`);
        }

        // flat 1D array / typed array
        if (Array.isArray(board) || ArrayBuffer.isView(board)) {
            // 2D?
            if (Array.isArray(board) && Array.isArray(board[0])) {
                if (board.length !== ROWS || board[0].length !== COLS) {
                    throw new Error(`Unsupported 2D board shape ${board.length}x${board[0].length}, expected ${ROWS}x${COLS}.`);
                }

                // assume row 0 is bottom (your convention)
                const bf = new Array(N);
                for (let y = 0; y < ROWS; y++) {
                    for (let x = 0; x < COLS; x++) bf[y * COLS + x] = board[y][x] | 0;
                }

                // if it doesn't look gravity-valid, try top-first as fallback
                if (!this._looksGravityValid(bf)) {
                    const tf = new Array(N);
                    for (let y = 0; y < ROWS; y++) {
                        for (let x = 0; x < COLS; x++) tf[y * COLS + x] = board[ROWS - 1 - y][x] | 0;
                    }
                    if (this._looksGravityValid(tf)) return tf;
                }
                return bf;
            }

            // 1D bottom-first row-major
            if (board.length === N) return board;

            // if someone passes 49 as a flat array, accept and drop the sentinel row too
            if (board.length === N_PAD) {
                return (typeof board.subarray === "function") ? board.subarray(0, N) : board.slice(0, N);
            }
        }

        throw new Error("Unsupported board format for bitboard parsing.");
    }


    // Quick sanity: in each column, once you hit an empty cell, you shouldn't see stones above it.
    _looksGravityValid(flat) {
        const ROWS = this.ROWS, COLS = this.COLS;
        for (let x = 0; x < COLS; x++) {
            let seenEmpty = false;
            for (let y = 0; y < ROWS; y++) {
                const v = flat[y * COLS + x] | 0;
                if (v === 0) seenEmpty = true;
                else if (seenEmpty) return false;
            }
        }
        return true;
    }


    _role_first_from_center(mask, p1, p2) {
        const b_d1 = 1n << BigInt(Connect4LookaheadJS.CENTER_COL * Connect4LookaheadJS.STRIDE + 0);
        if ((mask & b_d1) === 0n) return { roleFirstMark: 1, parityEnabled: false };
        if ((p1 & b_d1) !== 0n) return { roleFirstMark: 1, parityEnabled: true };
        return { roleFirstMark: -1, parityEnabled: true };
    }

    // ---------------- Internals: bit ops ----------------

    _popcount(x) {
        // Kernighan on BigInt
        let n = 0;
        while (x) {
            x &= (x - 1n);
            n++;
        }
        return n;
    }

    _has_won(bb) {
        const STRIDE = BigInt(Connect4LookaheadJS.STRIDE);

        let m = bb & (bb >> 1n);
        if ((m & (m >> 2n)) !== 0n) return true;

        m = bb & (bb >> STRIDE);
        if ((m & (m >> (2n * STRIDE))) !== 0n) return true;

        m = bb & (bb >> (STRIDE + 1n));
        if ((m & (m >> (2n * (STRIDE + 1n)))) !== 0n) return true;

        m = bb & (bb >> (STRIDE - 1n));
        if ((m & (m >> (2n * (STRIDE - 1n)))) !== 0n) return true;

        return false;
    }

    _can_play(mask, c) {
        return (mask & Connect4LookaheadJS.TOP_MASK[c]) === 0n;
    }

    _play_bit(mask, c) {
        return (mask + Connect4LookaheadJS.BOTTOM_MASK[c]) & Connect4LookaheadJS.COL_MASK[c];
    }

    _is_winning_move(pos, mask, c) {
        const mv = this._play_bit(mask, c);
        return this._has_won(pos | mv);
    }

    _count_immediate_wins_bits(pos, mask) {
        let cnt = 0;
        for (const c of Connect4LookaheadJS._CENTER_ORDER) {
            if (this._can_play(mask, c) && this._is_winning_move(pos, mask, c)) cnt++;
        }
        return cnt;
    }

    _has_any_immediate_win_bits(pos, mask) {
        for (const c of Connect4LookaheadJS._CENTER_ORDER) {
            if (this._can_play(mask, c) && this._is_winning_move(pos, mask, c)) return true;
        }
        return false;
    }

    _has_two_immediate_wins_bits(pos, mask) {
        let cnt = 0;
        for (const c of Connect4LookaheadJS._CENTER_ORDER) {
            if (this._can_play(mask, c) && this._is_winning_move(pos, mask, c)) {
                cnt++;
                if (cnt >= 2) return true;
            }
        }
        return false;
    }

    _is_immediate_blunder(pos, mask, c) {
        // after I play c, does opponent have a win-in-1?
        const mv = this._play_bit(mask, c);
        const nm = mask | mv;
        const oppPos = nm ^ (pos | mv);
        for (const cc of Connect4LookaheadJS._CENTER_ORDER) {
            if (this._can_play(nm, cc) && this._is_winning_move(oppPos, nm, cc)) return true;
        }
        return false;
    }

    _count_safe_moves(pos, mask) {
        let s = 0;
        for (const c of Connect4LookaheadJS._CENTER_ORDER) {
            if (this._can_play(mask, c) && !this._is_immediate_blunder(pos, mask, c)) s++;
        }
        return s;
    }

    _opp_has_double_threat_after_my_move(pos, mask, c) {
        const mv = this._play_bit(mask, c);
        const nm = mask | mv;
        const my_after = pos | mv;
        if (this._has_won(my_after)) return false;
        const opp_after = nm ^ my_after;
        return this._has_two_immediate_wins_bits(opp_after, nm);
    }

    _opp_can_reply_create_double_threat(pos, mask, c) {
        const mv = this._play_bit(mask, c);
        const nm = mask | mv;
        const my_after = pos | mv;
        if (this._has_won(my_after)) return false;

        const opp_pos = nm ^ my_after;

        for (const oc of Connect4LookaheadJS._CENTER_ORDER) {
            if (!this._can_play(nm, oc)) continue;

            const mv2 = this._play_bit(nm, oc);
            const nm2 = nm | mv2;
            const opp_after = opp_pos | mv2;

            // immediate win for opp
            if (this._has_won(opp_after)) return true;

            // if I have an immediate win after their reply, they won't choose this line (guard logic)
            if (this._has_any_immediate_win_bits(my_after, nm2)) continue;

            // if their reply creates >=2 immediate wins, danger
            if (this._has_two_immediate_wins_bits(opp_after, nm2)) return true;
        }

        return false;
    }

    // Root tactical helpers (fork + true win-in-2)
    _threat_count_after_move(pos, mask, c) {
        const mv = this._play_bit(mask, c);
        const nm = mask | mv;
        const my_after = pos | mv;
        const opp_after = nm ^ my_after;

        // if opp has any immediate win, this move is illegal-ish for threat counting
        if (this._count_immediate_wins_bits(opp_after, nm) !== 0) return -1;
        return this._count_immediate_wins_bits(my_after, nm);
    }

    _find_fork_move_root(pos, mask, legal) {
        let bestC = -1;
        let bestT = 1;
        for (let i = 0; i < legal.length; i++) {
            const c = legal[i];
            const t = this._threat_count_after_move(pos, mask, c);
            if (t >= 2 && t > bestT) {
                bestT = t;
                bestC = c;
            }
        }
        return bestC;
    }

    _is_forced_win_in_2_bits(pos, mask, c) {
        const mv = this._play_bit(mask, c);
        const nm = mask | mv;
        const my_after = pos | mv;
        const opp_after = nm ^ my_after;

        // if opp has immediate win now, not a win-in-2
        if (this._count_immediate_wins_bits(opp_after, nm) !== 0) return false;

        let anyReply = false;

        for (const oc of Connect4LookaheadJS._CENTER_ORDER) {
            if (!this._can_play(nm, oc)) continue;
            anyReply = true;

            const mv2 = this._play_bit(nm, oc);
            const nm2 = nm | mv2;

            // do I have a win-in-1 after this reply?
            let win1 = false;
            for (const cc of Connect4LookaheadJS._CENTER_ORDER) {
                if (this._can_play(nm2, cc) && this._is_winning_move(my_after, nm2, cc)) {
                    win1 = true;
                    break;
                }
            }
            if (!win1) return false;
        }

        return anyReply;
    }

    _find_forced_win_in_2_move_root(pos, mask, legal) {
        for (let i = 0; i < legal.length; i++) {
            const c = legal[i];
            if (this._is_forced_win_in_2_bits(pos, mask, c)) return c;
        }
        return -1;
    }

    // ---------------- Internals: evaluation ----------------

    _makeWArr() {
        // warr[k] for k=0..4
        const arr = new Float64Array(Connect4LookaheadJS.K + 1);
        arr[0] = 0.0;
        arr[1] = 0.0;
        arr[2] = Number(this.weights.get(2) || 0.0);
        arr[3] = Number(this.weights.get(3) || 0.0);
        arr[4] = Number(this.weights.get(4) || 0.0);
        return arr;
    }

    _evaluate(pos, mask, WARR, parity_enabled, root_pos_is_first, ply) {
        const opp = mask ^ pos;

        // column heights H[c] = popcount(mask & COL_MASK[c])
        const H = new Int16Array(Connect4LookaheadJS.COLS);
        for (let c = 0; c < Connect4LookaheadJS.COLS; c++) {
            H[c] = this._popcount(mask & Connect4LookaheadJS.COL_MASK[c]);
        }

        let score = 0.0;

        const W = Connect4LookaheadJS.WIN_MASKS.length;
        for (let idx = 0; idx < W; idx++) {
            const wmask = Connect4LookaheadJS.WIN_MASKS[idx];
            const mo = wmask & opp;
            const mp = wmask & pos;

            if (mo !== 0n && mp !== 0n) continue;

            const p = this._popcount(mp);
            const o = this._popcount(mo);
            if ((p + o) < 2) continue;

            let mul = 1.0;
            let ready_vertical3 = false;

            // per-empty floating penalty (correct model)
            if (p === 0 || o === 0) {
                const base = idx * 4;
                for (let k2 = 0; k2 < 4; k2++) {
                    const b = Connect4LookaheadJS.WIN_B[base + k2];
                    if ((mask & b) === 0n) {
                        const cc = Connect4LookaheadJS.WIN_C[base + k2] | 0;
                        const rr = Connect4LookaheadJS.WIN_R[base + k2] | 0;
                        const dh = (rr - H[cc]) | 0;

                        if (dh === 1) mul *= this.FLOATING_NEAR;
                        else if (dh >= 2) mul *= this.FLOATING_FAR;
                        else {
                            if (Connect4LookaheadJS.WIN_KIND[idx] === 1 && p === 3 && o === 0) {
                                ready_vertical3 = true;
                            }
                        }
                    }
                }
            }

            if (Connect4LookaheadJS.WIN_KIND[idx] === 1) mul *= this.VERT_MUL;

            if (o === 0) {
                score += mul * (p <= 4 ? WARR[p] : 0.0);
                if (ready_vertical3) score += this.VERT_3_READY_BONUS;
            } else if (p === 0) {
                score -= this.DEFENSIVE * mul * (o <= 4 ? WARR[o] : 0.0);
            }
        }

        const my_imm = this._count_immediate_wins_bits(pos, mask);
        const opp_imm = this._count_immediate_wins_bits(opp, mask);
        score += this.immediate_w * (my_imm - this.DEFENSIVE * opp_imm);

        if (my_imm >= 2) score += this.fork_w * (my_imm - 1);
        if (opp_imm >= 2) score -= this.DEFensIVE_SAFE() * (this.fork_w * (opp_imm - 1));

        // center bonus
        score += this.CENTER_BONUS * (
            this._popcount(pos & Connect4LookaheadJS.CENTER_MASK) -
            this._popcount(opp & Connect4LookaheadJS.CENTER_MASK)
        );

        // parity bonus (same logic as in Python)
        if (parity_enabled) {
            const is_root_turn = ((ply & 1) === 0);
            const pos_is_first = is_root_turn ? (root_pos_is_first === 1) : (root_pos_is_first !== 1);

            if (pos_is_first) {
                score += this.PARITY_BONUS * (
                    this._popcount(pos & Connect4LookaheadJS.ODD_MASK) -
                    this.DEFENSIVE * this._popcount(opp & Connect4LookaheadJS.EVEN_MASK)
                );
            } else {
                score += this.PARITY_BONUS * (
                    this._popcount(pos & Connect4LookaheadJS.EVEN_MASK) -
                    this.DEFENSIVE * this._popcount(opp & Connect4LookaheadJS.ODD_MASK)
                );
            }
        }

        // tempo (safe moves)
        if (this.TEMPO_W !== 0.0) {
            const my_safe = this._count_safe_moves(pos, mask);
            const opp_safe = this._count_safe_moves(opp, mask);
            score += this.TEMPO_W * (my_safe - this.DEFENSIVE * opp_safe);
        }

        return score;
    }

    // Small helper to avoid typo footguns in one place
    DEFensIVE_SAFE() { return this.DEFENSIVE; }

    // ---------------- Internals: search (TT, ordering, negamax) ----------------

    _makeSearchContext(WARR) {
        // TT arrays
        const TT_SIZE = 1 << C4LA_DEFAULTS.TT_SIZE_POW2;
        const TT_pos = new BigUint64Array(TT_SIZE);
        const TT_mask = new BigUint64Array(TT_SIZE);
        const TT_depth = new Int16Array(TT_SIZE);
        const TT_flag = new Int8Array(TT_SIZE);
        const TT_val = new Float64Array(TT_SIZE);
        const TT_move = new Int8Array(TT_SIZE);

        for (let i = 0; i < TT_SIZE; i++) TT_depth[i] = -1;

        // killers and history
        const killers = Array.from({ length: 64 }, () => [-1, -1]);
        const history = new Int32Array(Connect4LookaheadJS.COLS);

        // node counter
        const node_counter = { n: 0 };
        const max_nodes = Number.POSITIVE_INFINITY;

        return {
            WARR,
            killers,
            history,
            TT_SIZE,
            TT_pos,
            TT_mask,
            TT_depth,
            TT_flag,
            TT_val,
            TT_move,
            node_counter,
            max_nodes,
        };
    }

    _tt_hash(pos, mask, sizeMask) {
        // 64-bit mix, then take low bits
        const C = 0x9E3779B97F4A7C15n;
        let h = (pos ^ (mask * C)) & 0xFFFFFFFFFFFFFFFFn;
        h ^= (h >> 7n);
        return Number(h & BigInt(sizeMask));
    }

    _tt_lookup(pos, mask, depth, alpha, beta, ctx) {
        const sizeMask = (ctx.TT_SIZE - 1) | 0;
        let idx = this._tt_hash(pos, mask, sizeMask);

        for (let probes = 0; probes < C4LA_DEFAULTS.TT_MAX_PROBES; probes++) {
            const d = ctx.TT_depth[idx];
            if (d === -1) return { hit: false, val: 0.0, mv: -1, alpha, beta };

            if (ctx.TT_pos[idx] === pos && ctx.TT_mask[idx] === mask) {
                const flag = ctx.TT_flag[idx];  // 0 EXACT, 1 LOWER, 2 UPPER
                const val = ctx.TT_val[idx];
                const mv = ctx.TT_move[idx];

                if (d >= depth) {
                    if (flag === 0) return { hit: true, val, mv, alpha, beta };
                    if (flag === 1 && val > alpha) alpha = val;
                    else if (flag === 2 && val < beta) beta = val;
                    if (alpha >= beta) return { hit: true, val, mv, alpha, beta };
                }
                return { hit: false, val, mv, alpha, beta };
            }

            idx = (idx + 1) & sizeMask;
        }

        return { hit: false, val: 0.0, mv: -1, alpha, beta };
    }

    _tt_store(pos, mask, depth, val, alpha0, beta, best_mv, ctx) {
        let flag = 0; // EXACT
        if (val <= alpha0) flag = 2;       // UPPER
        else if (val >= beta) flag = 1;    // LOWER

        const sizeMask = (ctx.TT_SIZE - 1) | 0;
        let idx = this._tt_hash(pos, mask, sizeMask);

        let victim = -1;
        for (let probes = 0; probes < C4LA_DEFAULTS.TT_MAX_PROBES; probes++) {
            const d = ctx.TT_depth[idx];
            if (d === -1) { victim = idx; break; }
            if (ctx.TT_pos[idx] === pos && ctx.TT_mask[idx] === mask) { victim = idx; break; }
            if (victim === -1 || ctx.TT_depth[idx] < ctx.TT_depth[victim]) victim = idx;
            idx = (idx + 1) & sizeMask;
        }
        if (victim === -1) victim = idx;

        ctx.TT_pos[victim] = pos;
        ctx.TT_mask[victim] = mask;
        ctx.TT_depth[victim] = depth;
        ctx.TT_flag[victim] = flag;
        ctx.TT_val[victim] = val;
        ctx.TT_move[victim] = best_mv;
    }

    _order_moves(mask, ply, killers, history) {
        const moves = [];
        const scores = [];
        const k1 = killers[ply][0] | 0;
        const k2 = killers[ply][1] | 0;

        for (const c of Connect4LookaheadJS._CENTER_ORDER) {
            if (!this._can_play(mask, c)) continue;

            let s = 0;
            if (c === k1) s += 1000000;
            else if (c === k2) s += 500000;
            s += history[c] | 0;

            moves.push(c);
            scores.push(s);
        }

        // insertion sort descending by scores
        for (let i = 1; i < moves.length; i++) {
            const mv = moves[i];
            const sc = scores[i];
            let j = i - 1;
            while (j >= 0 && scores[j] < sc) {
                moves[j + 1] = moves[j];
                scores[j + 1] = scores[j];
                j--;
            }
            moves[j + 1] = mv;
            scores[j + 1] = sc;
        }

        return moves;
    }

    _negamax(pos, mask, depth, alpha, beta, ply, root_pos_is_first, parity_enabled,
        double_guard, fork_guard, ctx) {

        // node stop (kept, default infinite)
        ctx.node_counter.n++;
        if (ctx.node_counter.n >= ctx.max_nodes) {
            const v = this._evaluate(pos, mask, ctx.WARR, parity_enabled, root_pos_is_first, ply);
            return { val: v, mv: -1 };
        }

        const alpha0 = alpha;

        // TT lookup
        const tt = this._tt_lookup(pos, mask, depth, alpha, beta, ctx);
        alpha = tt.alpha; beta = tt.beta;
        if (tt.hit) return { val: tt.val, mv: tt.mv };

        // win-in-1 for side to move
        for (const c of Connect4LookaheadJS._CENTER_ORDER) {
            if (this._can_play(mask, c) && this._is_winning_move(pos, mask, c)) {
                return { val: (this.MATE_SCORE - ply), mv: c };
            }
        }

        // draw
        if (mask === Connect4LookaheadJS.FULL_MASK) return { val: 0.0, mv: -1 };

        // leaf
        if (depth === 0) {
            const v = this._evaluate(pos, mask, ctx.WARR, parity_enabled, root_pos_is_first, ply);
            return { val: v, mv: -1 };
        }

        let best_val = -1e100;
        let best_col = -1;

        const ordered = this._order_moves(mask, ply, ctx.killers, ctx.history);

        // prefer safe moves if any
        const safe = [];
        for (let i = 0; i < ordered.length; i++) {
            const c = ordered[i];
            if (!this._is_immediate_blunder(pos, mask, c)) safe.push(c);
        }
        let useMoves = safe.length ? safe : ordered;

        // hard guards (cheap prune)
        if (double_guard) {
            const tmp = [];
            for (let i = 0; i < useMoves.length; i++) {
                const c = useMoves[i];
                if (!this._opp_has_double_threat_after_my_move(pos, mask, c)) tmp.push(c);
            }
            if (tmp.length) useMoves = tmp;
        }

        if (fork_guard) {
            const tmp2 = [];
            for (let i = 0; i < useMoves.length; i++) {
                const c = useMoves[i];
                if (!this._opp_can_reply_create_double_threat(pos, mask, c)) tmp2.push(c);
            }
            if (tmp2.length) useMoves = tmp2;
        }

        for (let i = 0; i < useMoves.length; i++) {
            const c = useMoves[i];

            const mv = this._play_bit(mask, c);
            const nm = mask | mv;
            const next_pos = nm ^ (pos | mv);   // opponent-to-move

            const child = this._negamax(
                next_pos, nm,
                (depth - 1) | 0,
                -beta, -alpha,
                (ply + 1) | 0,
                root_pos_is_first,
                parity_enabled,
                double_guard,
                fork_guard,
                ctx
            );

            const val = -child.val;

            if (val > best_val) {
                best_val = val;
                best_col = c;
            }
            if (best_val > alpha) alpha = best_val;

            if (alpha >= beta) {
                // killer/history update
                if (ctx.killers[ply][0] !== c) {
                    ctx.killers[ply][1] = ctx.killers[ply][0];
                    ctx.killers[ply][0] = c;
                }
                ctx.history[c] += (depth * depth) | 0;
                break;
            }
        }

        this._tt_store(pos, mask, depth, best_val, alpha0, beta, best_col, ctx);
        return { val: best_val, mv: best_col };
    }

    _root_select_fixed(pos, mask, depth, root_pos_is_first, parity_enabled,
        double_guard, fork_guard, ctx) {

        // build legal list in center order
        let legal = [];
        for (const c of Connect4LookaheadJS._CENTER_ORDER) {
            if (this._can_play(mask, c)) legal.push(c);
        }
        if (!legal.length) return -1;

        // immediate win
        for (let i = 0; i < legal.length; i++) {
            const c = legal[i];
            if (this._is_winning_move(pos, mask, c)) return c | 0;
        }

        // single must-block
        const opp_pos = mask ^ pos;
        let block_col = -1, block_count = 0;
        for (let i = 0; i < legal.length; i++) {
            const c = legal[i];
            if (this._is_winning_move(opp_pos, mask, c)) {
                block_col = c;
                block_count++;
                if (block_count > 1) break;
            }
        }
        if (block_count === 1) return block_col | 0;

        // avoid obvious handovers if possible
        const safe = [];
        for (let i = 0; i < legal.length; i++) {
            const c = legal[i];
            if (!this._is_immediate_blunder(pos, mask, c)) safe.push(c);
        }
        if (safe.length) legal = safe;

        // root guards
        if (double_guard) {
            const tmp = [];
            for (let i = 0; i < legal.length; i++) {
                const c = legal[i];
                if (!this._opp_has_double_threat_after_my_move(pos, mask, c)) tmp.push(c);
            }
            if (tmp.length) legal = tmp;
        }

        if (fork_guard) {
            const tmp2 = [];
            for (let i = 0; i < legal.length; i++) {
                const c = legal[i];
                if (!this._opp_can_reply_create_double_threat(pos, mask, c)) tmp2.push(c);
            }
            if (tmp2.length) legal = tmp2;
        }

        // root tactical pre-pass (optional)
        if (this.WIN2_CHECK) {
            const fork_mv = this._find_fork_move_root(pos, mask, legal);
            if (fork_mv !== -1) return fork_mv | 0;

            const win2_mv = this._find_forced_win_in_2_move_root(pos, mask, legal);
            if (win2_mv !== -1) return win2_mv | 0;
        }

        // heights for parity bias
        const H = new Int16Array(Connect4LookaheadJS.COLS);
        for (let c0 = 0; c0 < Connect4LookaheadJS.COLS; c0++) {
            H[c0] = this._popcount(mask & Connect4LookaheadJS.COL_MASK[c0]);
        }

        const root_is_first = (root_pos_is_first === 1);
        const pref_parity_root = root_is_first ? 0 : 1; // row parity preference: first wants odd rows (r%2==0)
        const pref_parity_opp = root_is_first ? 1 : 0;

        let best_move = legal[0] | 0;
        let best_val = -1e100;

        for (let i = 0; i < legal.length; i++) {
            const c = legal[i] | 0;
            let bias = 0.0;

            const r = H[c] | 0;

            if (parity_enabled) {
                bias += (((r & 1) === pref_parity_root) ? this.PARITY_MOVE_W : -this.PARITY_MOVE_W);
                const r2 = (r + 1) | 0;
                if (r2 < Connect4LookaheadJS.ROWS) {
                    if ((r2 & 1) === pref_parity_opp) bias -= this.PARITY_UNLOCK_W;
                }
            }

            const mv = this._play_bit(mask, c);
            const nm = mask | mv;
            const my_after = pos | mv;
            const opp_after = nm ^ my_after;

            if (this.THREATSPACE_W !== 0.0) {
                const my_threats = this._count_immediate_wins_bits(my_after, nm);
                bias += this.THREATSPACE_W * my_threats;

                const opp_threats = this._count_immediate_wins_bits(opp_after, nm);
                bias -= this.DEFENSIVE * this.THREATSPACE_W * 0.25 * opp_threats;
            }

            // search from opponent-to-move after my move, then negate
            const child = this._negamax(
                opp_after, nm,
                (depth - 1) | 0,
                -this.MATE_SCORE, this.MATE_SCORE,
                1,
                root_pos_is_first,
                parity_enabled,
                double_guard,
                fork_guard,
                this._makeSearchContext(ctx.WARR) // root: fresh TT/killers per move for stability
            );

            const val = -child.val + bias;

            if (val > best_val) {
                best_val = val;
                best_move = c;
            }
        }

        return best_move | 0;
    }
}
