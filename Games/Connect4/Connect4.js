/*jshint browser: true */
/*jshint -W097 */
/*jshint -W117 */
/*jshint -W061 */
"use strict";

/////////////////////////////////////////////////
/*
      
TODO:
    * 
known bugs: 
    * i don't do bugs

retests:

engine changes:



 */
////////////////////////////////////////////////////

const DEBUG = {
    SETTING: true,
    FPS: false,
    VERBOSE: true,
    max17: false,
    keys: false,
    simulation: false,
    drawToConsole: false,

    board: [
        0, 2, 1, 2, 2, 2, 0,
        0, 1, 1, 1, 2, 0, 0,
        0, 2, 1, 2, 0, 0, 0,
        0, 1, 2, 2, 0, 0, 0,
        0, 0, 1, 1, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0,
    ],
};

const INI = {
    SCREEN_BORDER: 196,
    ROWS: 6,
    COLS: 7,
    INROW: 4,
    GRIDSIZE: 100,
    LEFT_X: -1,
    RADIUS_FACTOR: 0.4,
    RADIUS: null,
    OVER_TURN: 6 * 7 + 1,
    MIN_END_TURN: 4 + 3,
    INROW2: 1,
    INROW3: 100,
    INROW4: 10000,
    IMMEDIATE_WIN: 500,
    FORK_BONUS: 150,
    DEFENSIVE_FACTOR: 1.5,
    FLOATING_NEAR: 0.50, //0.50
    FLOATING_FAR: 0.25, //0.25
};

const PRG = {
    VERSION: "1.3.7",
    NAME: "Connect-4",
    YEAR: "2025",
    SG: null,
    CSS: "color: #239AFF;",
    INIT() {
        console.log("%c**************************************************************************************************************************************", PRG.CSS);
        console.log(`${PRG.NAME} ${PRG.VERSION} by Lovro Selic, (c) LaughingSkull ${PRG.YEAR} on ${navigator.userAgent}`);
        console.log("%c**************************************************************************************************************************************", PRG.CSS);
        $("#title").html(PRG.NAME);
        $("#version").html(`${PRG.NAME} V${PRG.VERSION} <span style='font-size:14px'>&copy</span> LaughingSkull ${PRG.YEAR}`);
        $("input#toggleAbout").val("About " + PRG.NAME);
        $("#about fieldset legend").append(" " + PRG.NAME + " ");

        ENGINE.autostart = true;
        ENGINE.start = PRG.start;
        ENGINE.readyCall = GAME.setup;
        ENGINE.setGridSize(64);
        ENGINE.setSpriteSheetSize(64);
        ENGINE.init();
    },
    setup() {
        if (DEBUG.SETTING) {
            $("#engine_version").html(ENGINE.VERSION);
            $("#grid_version").html(GRID.VERSION);
            $("#lib_version").html(LIB.VERSION);
        } else {
            $('#debug').hide();
        }

        $("#toggleAbout").click(function () {
            $("#about").toggle(400);
        });
        $("#toggleVersion").click(function () {
            $("#debug").toggle(400);
        });


        //boxes
        ENGINE.gameWIDTH = 1024;
        ENGINE.titleWIDTH = ENGINE.gameWIDTH + 2 * INI.SCREEN_BORDER;
        ENGINE.sideWIDTH = ENGINE.titleWIDTH - ENGINE.gameWIDTH - INI.SCREEN_BORDER;
        ENGINE.gameHEIGHT = 768;
        ENGINE.titleHEIGHT = 96;
        ENGINE.bottomHEIGHT = 80;
        ENGINE.bottomWIDTH = ENGINE.titleWIDTH;

        $("#bottom").css("margin-top", ENGINE.gameHEIGHT + ENGINE.titleHEIGHT + ENGINE.bottomHEIGHT);
        $(ENGINE.gameWindowId).width(ENGINE.gameWIDTH + 2 * ENGINE.sideWIDTH + 4);
        ENGINE.addBOX("TITLE", ENGINE.titleWIDTH, ENGINE.titleHEIGHT, ["title"], null);
        ENGINE.addBOX("LSIDE", INI.SCREEN_BORDER, ENGINE.gameHEIGHT, ["Lsideback", "red"], "side");
        ENGINE.addBOX("ROOM", ENGINE.gameWIDTH, ENGINE.gameHEIGHT, ["background", "token", "grid", "front",
            "col_labels", "strike", "text", "FPS",
            "button", "click"], "side");
        ENGINE.addBOX("SIDE", ENGINE.sideWIDTH, ENGINE.gameHEIGHT, ["sideback", "blue",], "fside");
        ENGINE.addBOX("DOWN", ENGINE.bottomWIDTH, ENGINE.bottomHEIGHT, ["bottom", "bottomText", "subtitle"], null);

        /* Connect-4 overrides */
        ENGINE.setGridSize(INI.GRIDSIZE);
        MAPDICT.RED = 1;
        MAPDICT.BLUE = 2;
        MAPDICT.red = 1;
        MAPDICT.blue = 2;

        INI.LEFT_X = (ENGINE.gameWIDTH - INI.COLS * ENGINE.INI.GRIDPIX) / 2;
        INI.RADIUS = Math.round(INI.RADIUS_FACTOR * ENGINE.INI.GRIDPIX);

        /**¸DOM setup */

        $("#game_mode").on("click", () => {
            $("#number_of_runs").prop("disabled", true);
        });

        $("#analyze_mode").on("click", () => {
            $("#number_of_runs").prop("disabled", false);
        });

        /** dev settings */
        if (DEBUG.VERBOSE) {
            ENGINE.verbose = true;
        }
    },
    start() {
        console.log("%c**************************************************************************************************************************************", PRG.CSS);
        console.log(`${PRG.NAME} ${PRG.VERSION} STARTED!`);
        console.log("%c**************************************************************************************************************************************", PRG.CSS);
        $(ENGINE.topCanvas).off("mousemove", ENGINE.mouseOver);
        $(ENGINE.topCanvas).off("click", ENGINE.mouseClick);
        $(ENGINE.topCanvas).css("cursor", "");

        $("#startGame").addClass("hidden");
        ENGINE.disableDefaultKeys();
        TITLE.startTitle();
    }
};

const BOARD = {
    patterns: null,
    coordinates: null,
    drawFront() {
        let CTX = LAYER.front;
        const GS = ENGINE.INI.GRIDPIX;

        for (let x = 0; x < INI.COLS; x++) {
            for (let y = 0; y < INI.ROWS; y++) {
                let grid = new Grid(x, y);
                this.drawCoverItem(CTX, grid);
            }
        }
        //this.drawTopGrid(CTX, GS);
        this.drawColLabels(GS);
    },
    drawTopGrid(CTX, GS) {
        CTX.save();
        CTX.strokeStyle = "#333";
        CTX.setLineDash([1, 2]);
        let x = INI.LEFT_X;
        let y = ENGINE.gameHEIGHT - (INI.ROWS + 1) * GS;
        for (let i = 0; i < INI.COLS; i++) {
            CTX.beginPath();
            CTX.rect(x, y, GS, GS);
            CTX.stroke();
            x += GS;
        }
        CTX.restore();
    },
    drawColLabels(GS, move = null, player = null) {
        let CTX = LAYER.col_labels;
        ENGINE.clearLayer("col_labels");
        const GS2 = Math.floor(GS / 2);
        const fs = 42;
        CTX.font = `${fs}px CompSmooth`;
        CTX.textAlign = "center";
        CTX.fillStyle = "rgba(100, 100, 100, 0.3)";
        for (let x = 0; x < INI.COLS; x++) {
            const y = GS * 0.5;

            if (x === move) {
                CTX.fillStyle = player;
            } else CTX.fillStyle = "rgba(100, 100, 100, 0.3)";

            CTX.fillText(x + 1, x * GS + INI.LEFT_X + GS2, y);
        }
    },
    drawCoverItem(CTX, grid) {
        const GS = ENGINE.INI.GRIDPIX;
        let OFF = GS / 2;
        const [x, y] = this.gridToCoord(grid);
        CTX.save();
        CTX.translate(OFF, OFF);
        CTX.beginPath();                                                // Clipping path - only clip will be visible
        CTX.rect(x - OFF, y - OFF, GS, GS);                             // Outer rectangle
        CTX.arc(x, y, INI.RADIUS, 0, Math.PI * 2, true);                // Hole anticlockwise¸
        CTX.clip();
        CTX.fillStyle = "#228B22";                                      //background
        CTX.fillRect(x - OFF, y - OFF, GS, GS);
        CTX.restore();
    },
    gridToCoord(grid, offset = 0) {
        const GS = ENGINE.INI.GRIDPIX;
        let x = grid.x * GS + INI.LEFT_X + offset;
        let y = ENGINE.gameHEIGHT - (grid.y + 1) * GS + offset;
        return [x, y];
    },
    drawContent() {
        let CTX = LAYER.grid;
        for (let x = 0; x < INI.COLS; x++) {
            for (let y = 0; y < INI.ROWS; y++) {
                let grid = new Grid(x, y);
                this.drawCircle(CTX, grid);
            }
        }
    },
    drawCircle(CTX, grid) {
        if (GAME.map.isZero(grid)) return;
        let OFF = ENGINE.INI.GRIDPIX / 2;
        const value = GAME.map.getValue(grid);
        let color = null;

        if (value === MAPDICT.RED) {
            color = "#FF0000";
        } else color = "#0000FF";

        const [x, y] = this.gridToCoord(grid);
        CTX.fillStyle = color;
        CTX.save();
        CTX.translate(OFF, OFF);
        CTX.beginPath();
        CTX.arc(x, y, INI.RADIUS, 0, Math.PI * 2, false);
        CTX.fill();
        CTX.restore();
    },
    importBoard(list) {
        for (let i = 0; i < list.length; i++) {
            GAME.map.map[i] = list[i];
        }
    },

    /**
     * @param {*} players array of players to check, allows [1], [2], [1,2]
     */
    boardToPatterns(players, board = GAME.map) {
        const patterns = [];
        const coordinates = [];
        let pat, coord;
        let functs = ["boardDiagonals", "boardHorizontals", "boardVerticals"];
        for (const F of functs) {
            [pat, coord] = this[F](players, board);
            patterns.push(...pat);
            coordinates.push(...coord);
        }

        return [patterns, coordinates];
    },
    boardDiagonals(players, board) {
        const GA = board;
        const patterns = [];
        const coordinates = [];

        // / diagonals (bottom-left to top-right)
        for (let x = 0; x <= INI.COLS - INI.INROW; x++) {
            for (let y = 0; y <= INI.ROWS - INI.INROW; y++) {
                const pattern = [];
                const coord = [];

                for (let i = 0; i < INI.INROW; i++) {
                    const grid = new Grid(x + i, y + i);
                    pattern.push(GA.map[GA.gridToIndex(grid)]);
                    coord.push([grid.x, grid.y]);
                }

                if (this.isValidPattern(pattern, players)) {
                    patterns.push(pattern);
                    coordinates.push(coord);
                }
            }
        }

        // \ diagonals (bottom-right to top-left)
        for (let x = INI.INROW - 1; x < INI.COLS; x++) {
            for (let y = 0; y <= INI.ROWS - INI.INROW; y++) {
                const pattern = [];
                const coord = [];

                for (let i = 0; i < INI.INROW; i++) {
                    const grid = new Grid(x - i, y + i);
                    pattern.push(GA.map[GA.gridToIndex(grid)]);
                    coord.push([grid.x, grid.y]);
                }

                if (this.isValidPattern(pattern, players)) {
                    patterns.push(pattern);
                    coordinates.push(coord);
                }
            }
        }

        return [patterns, coordinates];
    },
    boardHorizontals(players, board) {
        const GA = board;
        const patterns = [];
        const coordinates = [];

        for (let x = 0; x <= INI.COLS - INI.INROW; x++) {
            for (let y = 0; y < INI.ROWS; y++) {
                const pattern = [];
                const coord = [];

                for (let i = 0; i < INI.INROW; i++) {
                    const grid = new Grid(x + i, y);
                    pattern.push(GA.map[GA.gridToIndex(grid)]);
                    coord.push([grid.x, grid.y]);
                }

                if (this.isValidPattern(pattern, players)) {
                    patterns.push(pattern);
                    coordinates.push(coord);
                }
            }
        }

        return [patterns, coordinates];
    },
    boardVerticals(players, board) {
        const GA = board;
        const patterns = [];
        const coordinates = [];
        for (let x = 0; x < INI.COLS; x++) {
            for (let y = 0; y <= INI.ROWS - INI.INROW; y++) {
                const pattern = [];
                const coord = [];

                for (let i = 0; i < INI.INROW; i++) {
                    const grid = new Grid(x, y + i);
                    pattern.push(GA.map[GA.gridToIndex(grid)]);
                    coord.push([grid.x, grid.y]);
                }

                if (this.isValidPattern(pattern, players)) {
                    patterns.push(pattern);
                    coordinates.push(coord);
                }
            }
        }

        return [patterns, coordinates];
    },
    isValidPattern(pattern, players) {
        const zeros = pattern.filter(v => v === 0).length;
        const matches = pattern.filter(v => players.includes(v)).length;

        return (
            zeros < 3 &&
            matches >= 2
        );
    },
    countWindowsInPattern(patterns, num, player) {
        const indices = [];
        patterns.forEach((window, index) => {
            const playerCount = window.filter(v => v === player).length;
            const zeroCount = window.filter(v => v === 0).length;
            if (playerCount === num && zeroCount === INI.INROW - num) {
                indices.push(index);
            }
        });

        return { count: indices.length, indices: indices };
    },
    printBoardToConsole(GA) {
        const cellChar = (val) => {
            if (val === 1) return 'X';                  //red
            if (val === 2) return 'O';                  //blue
            return ' ';
        };

        if (!DEBUG.simulation) return;

        const cols = INI.COLS;
        const rows = INI.ROWS;
        const grid = GA.map;

        // Column header
        let header = '    ';
        for (let c = 0; c < cols; c++) {
            header += ` ${c + 1}  `;
        }
        console.log(header);

        // Build each row from top to bottom
        for (let y = rows - 1; y >= 0; y--) {
            let horizontalBorder = '   +' + '---+'.repeat(cols);
            let rowContent = `${(y + 1).toString().padStart(2)} |`;

            for (let x = 0; x < cols; x++) {
                const val = grid[GA.gridToIndex(new Grid(x, y))];
                rowContent += ` ${cellChar(val)} |`;
            }

            console.log(horizontalBorder);
            console.log(rowContent);
        }

        // Bottom border
        console.log('   +' + '---+'.repeat(cols));
    },

    // --- fast, precomputed 4-in-a-row windows + value helper ---
    _ALL_WINDOWS: null,
    _ensureAllWindows() {
        if (BOARD._ALL_WINDOWS) return;

        const W = [];
        // horizontals
        for (let y = 0; y < INI.ROWS; y++) {
            for (let x = 0; x <= INI.COLS - INI.INROW; x++) {
                W.push([[x, y], [x + 1, y], [x + 2, y], [x + 3, y]]);
            }
        }
        // verticals
        for (let x = 0; x < INI.COLS; x++) {
            for (let y = 0; y <= INI.ROWS - INI.INROW; y++) {
                W.push([[x, y], [x, y + 1], [x, y + 2], [x, y + 3]]);
            }
        }
        // diag up-right (/ from bottom-left to top-right)
        for (let x = 0; x <= INI.COLS - INI.INROW; x++) {
            for (let y = 0; y <= INI.ROWS - INI.INROW; y++) {
                W.push([[x, y], [x + 1, y + 1], [x + 2, y + 2], [x + 3, y + 3]]);
            }
        }
        // diag up-left (\ from bottom-right to top-left)
        for (let x = INI.INROW - 1; x < INI.COLS; x++) {
            for (let y = 0; y <= INI.ROWS - INI.INROW; y++) {
                W.push([[x, y], [x - 1, y + 1], [x - 2, y + 2], [x - 3, y + 3]]);
            }
        }

        BOARD._ALL_WINDOWS = W;
    },

    _val(GA, x, y) {
        return GA.map[GA.gridToIndex(new Grid(x, y))];
    },
    findFourIndices(GA, player) {
        const out = [];
        for (let i = 0; i < BOARD._ALL_WINDOWS.length; i++) {
            const w = BOARD._ALL_WINDOWS[i];
            let ok = true;
            for (let k = 0; k < 4; k++) {
                const [x, y] = w[k];
                if (BOARD._val(GA, x, y) !== player) {
                    ok = false;
                    break;
                }
            }
            if (ok) out.push(i);
        }
        return out;
    },
    countPure(GA, player, n) {
        let cnt = 0;
        for (const w of BOARD._ALL_WINDOWS) {
            let p = 0, o = 0;
            for (let k = 0; k < 4; k++) {
                const v = BOARD._val(GA, w[k][0], w[k][1]);
                if (v === player) {
                    p++;
                } else if (v !== 0) o++;
            }
            if (o === 0 && p === n) cnt++;
        }
        return cnt;
    },

};

const AGENT = {
    Human() {
        let legal_moves = AGENT_MANAGER.getLegalMoves();
        legal_moves = legal_moves.map(i => i + 1);
        let move = TURN_MANAGER.getInput(legal_moves);
        return move;
    },
    Random() {
        let legal_moves = AGENT_MANAGER.getLegalMoves();
        return legal_moves.chooseRandom();
    },
    Silly(playerIndex) {
        return AGENT_MANAGER.N_step_lookahead(playerIndex, 3);
    },
    Smarty(playerIndex) {
        return AGENT_MANAGER.N_step_lookahead(playerIndex, 5);
    },
    Prophet(playerIndex) {
        return AGENT_MANAGER.N_step_lookahead(playerIndex, 7);
    }
};

const AGENT_MANAGER = {
    getLegalMoves(map = GAME.map) {
        const legalMoves = [];
        for (let c = 0; c < INI.COLS; c++) {
            let grid = new Grid(c, INI.ROWS - 1);
            if (map.isZero(grid)) legalMoves.push(c);
        }
        return legalMoves;
    },
    getLegalCentreOrderedMoves(map = GAME.map) {
        const legalMoves = [];
        for (let c = 0; c < INI.COLS; c++) {
            let grid = new Grid(TURN_MANAGER.order[c], INI.ROWS - 1);
            if (map.isZero(grid)) legalMoves.push(TURN_MANAGER.order[c]);
        }
        return legalMoves;
    },
    getEmptyRow(map, col) {
        for (let row = 0; row < INI.ROWS; row++) {
            let grid = new Grid(col, row);
            if (map.isZero(grid)) return grid;
        }
        throw "This should be only called for legal moves!!";
    },
    getDestination(move) {
        return this.getEmptyRow(GAME.map, move);
    },
    N_step_lookahead(playerIndex, N) {
        let moves = this.getLegalCentreOrderedMoves();
        const scores = {};
        for (const move of moves) {
            scores[move] = this.scoreMove(GAME.map, move, playerIndex, N);
        }
        const maxScore = Math.max(...Object.values(scores));
        const bestMoves = Object.entries(scores)
            .filter(([_, score]) => score === maxScore)
            .map(([move]) => parseInt(move));

        const innermost = this.innermost(bestMoves);
        const selectedMove = innermost.chooseRandom();

        return selectedMove;
    },
    scoreMove(grid, move, playerIndex, N) {
        const nextGrid_GA = this.dropPiece(grid, move, playerIndex);
        return this.minimax(nextGrid_GA, N - 1, false, playerIndex, -Infinity, Infinity);
    },
    dropPiece(grid, move, playerIndex) {
        let nextGrid = grid.clone();                                                                                        //this is GA!
        let placedGrid = this.getEmptyRow(nextGrid, move);                                                                  //filtered for valid moves
        nextGrid.setValue(placedGrid, playerIndex);
        if (DEBUG.drawToConsole) BOARD.printBoardToConsole(nextGrid);
        return nextGrid;
    },
    minimax(GA, depth, maximizingPlayer, playerIndex, A, B) {
        if (depth === 0 || this.isTerminalNode(GA)) {
            return this.getHeuristic(playerIndex, GA); // signature preserved
        }
        const validMoves = this.getLegalCentreOrderedMoves(GA);

        if (maximizingPlayer) {
            let value = -Infinity;
            for (const col of validMoves) {
                const childGA = this.dropPiece(GA, col, playerIndex);
                const newValue = this.minimax(childGA, depth - 1, false, playerIndex, A, B);
                value = Math.max(value, newValue);
                if (value >= B) break;
                A = Math.max(A, value);
            }
            return value;
        } else {
            let value = Infinity;
            const opponent = (playerIndex % 2) + 1;
            for (const col of validMoves) {
                const childGA = this.dropPiece(GA, col, opponent);
                const newValue = this.minimax(childGA, depth - 1, true, playerIndex, A, B);
                value = Math.min(value, newValue);
                if (value <= A) break;
                B = Math.min(B, value);
            }
            return value;
        }
    },
    // gravity-aware heuristic with soft discount for floating windows (no gravity support)
    getHeuristic(playerIndex, currentBoard) {
        const GA = currentBoard;
        const DEF = INI.DEFENSIVE_FACTOR;
        const NEAR = INI.FLOATING_NEAR;
        const FAR = INI.FLOATING_FAR;
        const opp = (playerIndex % 2) + 1;

        const w = [0.0, 0.0, Number(INI.INROW2), Number(INI.INROW3), Number(INI.INROW4)];

        // column heights (bottom-based y: 0..5). Fast and enough for 7x6.
        const heights = new Array(7);
        for (let x = 0; x < 7; x++) {
            let h = 0;
            while (h < 6 && BOARD._val(GA, x, h) !== 0) h++;
            heights[x] = h; // number of filled cells in column x
        }

        let score = 0.0;

        for (const win of BOARD._ALL_WINDOWS) {
            let p = 0, o = 0;
            let need = 0; // total fillers required across empties to make all playable

            for (let k = 0; k < 4; k++) {
                const x = win[k][0], y = win[k][1];
                const v = BOARD._val(GA, x, y);

                if (v === 0) {
                                                                // if y>0, cell is playable only if y-1 < heights[x]
                    if (y > 0) {
                        const deficit = y - heights[x];         // how many tokens must be dropped first
                        if (deficit > 0) need += deficit;
                    }
                } else if (v === playerIndex) {
                    p++;
                } else if (v === opp) {
                    o++;
                }
            }

            const mul = (need === 0) ? 1.0 : (need === 1 ? NEAR : FAR);

            if (o === 0) {
                score += mul * w[p];                // pure-us window
            } else if (p === 0) {
                score -= mul * DEF * w[o];          // pure-them window
            }
        }

        // Immediate wins & forks (exact via simulated drops)
        const myImm = this.countImmediateWins(GA, playerIndex).length;
        const oppImm = this.countImmediateWins(GA, opp).length;

        score += INI.IMMEDIATE_WIN * (myImm - DEF * oppImm);
        if (myImm >= 2) score += INI.FORK_BONUS * (myImm - 1);
        if (oppImm >= 2) score -= DEF * (INI.FORK_BONUS * (oppImm - 1));

        return Math.ceil(score);
    },
    hasFour(board, playerIndex) {
        for (const win of BOARD._ALL_WINDOWS) {
            let ok = true;
            for (let k = 0; k < 4; k++) {
                const [x, y] = win[k];
                if (BOARD._val(board, x, y) !== playerIndex) {
                    ok = false;
                    break;
                }
            }
            if (ok) return true;
        }
        return false;
    },
    countImmediateWins(board, playerIndex) {
        const wins = [];
        const legalColumns = this.getLegalMoves(board);
        for (let column of legalColumns) {
            const leaf_board = this.dropPiece(board, column, playerIndex);
            if (this.hasFour(leaf_board, playerIndex)) {
                wins.push(column);
            }
        }
        return wins;
    },
    isTerminalNode(GA) {
        if (this.hasFour(GA, 1) || this.hasFour(GA, 2)) return true;
        return this.getLegalMoves(GA).length === 0;
    },
    innermost(arr) {
        const mid = (INI.COLS - 1) / 2;
        let bestMoves = [];
        let bestScore = -Infinity;

        for (const move of arr) {
            const score = -Math.abs(move - mid);
            if (score > bestScore) {
                bestScore = score;
                bestMoves = [move];                                 // new best
            } else if (score === bestScore) {
                bestMoves.push(move);                               // equally good
            }
        }

        return bestMoves;
    }
}

class Token {
    constructor(move, startGrid, destination, player) {
        this.move = move;
        this.moveState = new MoveState(startGrid, UP, GAME.map, FP_Grid.toClass(startGrid));
        this.destination = destination;
        this.player = player;
        this.onDestination = false;
        this.moveSpeed = parseInt($("#animation_speed")[0].value, 10);
    }
    manage(lapsedTime) {
        if (this.moveState.moving) {
            GRID.translateMoveState(this, lapsedTime, 1, -1);
        } else this.makeMove();
    }
    makeMove() {
        if (GRID.same(this.moveState.startGrid, this.destination)) {
            this.onDestination = true;
            return;
        }
        this.moveState.next(UP);
    }
}

const TURN_MANAGER = {
    analysis_in_progress: false,
    runCounter: 0,
    winner: null,
    patterns: null,
    indices: null,
    nextPlayerIndex: null,
    players: ["red", "blue"],
    playerPieces: {
        red: 1,
        blue: 2,
    },
    turn_completed: null,
    turn: null,
    agent: {
        red: null,
        blue: null,
    },
    name: {
        red: null,
        blue: null,
    },
    score: {
        red: null,
        blue: null,
    },
    mode: 1,                        //game (default)
    token: null,
    awaitingInput: false,
    lastInput: null,
    inputCallback: null,
    allowed: [],
    order: null,
    ANALYSIS: {},
    STARTS: {
        red: 0,
        blue: 0,
    },
    init() {
        BOARD._ensureAllWindows();
        const next = $("#select_player_start")[0].value;
        switch (next) {
            case "red":
                this.nextPlayerIndex = 0;
                break;
            case "blue":
                this.nextPlayerIndex = 1;
                break;
            case "random":
                this.nextPlayerIndex = RND(0, 1);
                break;

        }
        if ($('input[name="mode"]:checked').val() === "analyze") this.mode = 0;
        this.turn = 0;
        this.turn_completed = true;
        this.agent.red = $("#red_player_agents")[0].value;
        this.agent.blue = $("#blue_player_agents")[0].value;
        this.name.red = $("#red_player_name")[0].value;
        this.name.blue = $("#blue_player_name")[0].value;
        this.score.red = 0;
        this.score.blue = 0;
        this.winner = null;
        this.awaitingInput = false;
        this.lastInput = null;
        this.allowed = [];
        this.order = this.getCenterOutOrder();

        if (this.mode === 0) {
            if (!this.analysis_in_progress) this.initAnalysis();
            this.STARTS[this.players[this.nextPlayerIndex]]++;
        }
    },
    initAnalysis() {
        this.runCounter = parseInt($("#number_of_runs")[0].value, 10);
        this.analysis_in_progress = true;
        this.ANALYSIS = {};
        this.ANALYSIS[this.agent.red] = 0;
        this.ANALYSIS[this.agent.blue] = 0;
        this.ANALYSIS.Tie = 0;
        this.STARTS.red = 0;
        this.STARTS.blue = 0;
    },
    getCenterOutOrder(cols = INI.COLS) {
        const order = [];
        for (let i = 0; i < cols; i++) {
            if (i % 2 === 0) {
                order.push(Math.floor(cols / 2) + Math.floor(i / 2));
            } else {
                order.push(Math.floor(cols / 2) - Math.floor(i / 2) - 1);
            }
        }
        return order;
    },
    getPlayer() {
        let player = this.players[this.nextPlayerIndex];
        this.switchPlayer();
        return player;
    },
    playerToIndex(player) {
        return this.players.indexOf(player);
    },
    switchPlayer() {
        this.nextPlayerIndex++;
        this.nextPlayerIndex %= 2;
    },
    nextPlayer() {
        if (TURN_MANAGER.awaitingInput) return;

        let move = null;
        let player = null;

        if (this.lastInput != null) {
            this.switchPlayer();                                                                    //need to switch to remain the same!
            player = this.players[this.nextPlayerIndex];
            move = this.lastInput;
            this.lastInput = null;
            this.switchPlayer();
        } else {
            this.turn++;
            if (this.turn === INI.OVER_TURN) {
                this.winner = "Tie";
                GAME.completed = true;
                return;
            }

            player = this.getPlayer();
            move = AGENT[this.agent[player]](this.playerToIndex(player) + 1);
        }

        if (TURN_MANAGER.awaitingInput) {
            SUBTITLE.subtitle(`${this.name[player]}: waiting for input`, player);
            return;
        }

        this.turn_completed = false;
        const destination = AGENT_MANAGER.getDestination(move);

        if (this.mode) {
            this.setMove(move, destination, player);
        } else this.applyDestination(destination, player);

        const nextPlayer = this.name[this.players[(this.playerToIndex(player) + 1) % 2]];
        SUBTITLE.subtitle(`${nextPlayer}: thinking`, nextPlayer);
    },
    setMove(move, destination, player) {
        this.token = new Token(move, new Grid(move, INI.ROWS), destination, player);
    },
    applyDestination(destination, player) {
        this.turn_completed = true;
        this.token = null;

        // place the token and redraw
        GAME.map.setToken(destination, player);
        BOARD.drawContent();

        // win check (window-based)
        const pid = this.playerPieces[player];                // 1 or 2
        const winIdx = BOARD.findFourIndices(GAME.map, pid);  // [] or [indices...]
        if (winIdx.length) return this.gameCompleted(winIdx, player);

        // score (pure 2- and 3-in-a-rows)
        const inrow2 = BOARD.countPure(GAME.map, pid, 2);
        const inrow3 = BOARD.countPure(GAME.map, pid, 3);
        TURN_MANAGER.score[player] = inrow2 * INI.INROW2 + inrow3 * INI.INROW3;
        TITLE.score();

        // UI
        BOARD.drawColLabels(ENGINE.INI.GRIDPIX, destination.x, player);
    },
    manage(lapsedTime) {
        if (this.turn_completed) return this.nextPlayer();
        if (this.token) this.token.manage(lapsedTime);
        if (this.token.onDestination) {
            this.applyDestination(this.token.destination, this.token.player);
        }
    },
    drawToken() {
        if (!this.token) return;
        ENGINE.clearLayer("token");
        const CTX = LAYER.token;
        CTX.fillStyle = this.token.player;
        const [x, y] = BOARD.gridToCoord(this.token.moveState.pos);
        const OFF = ENGINE.INI.GRIDPIX / 2;
        CTX.save();
        CTX.translate(OFF, OFF);
        CTX.beginPath();
        CTX.arc(x, y, INI.RADIUS, 0, Math.PI * 2, false);
        CTX.fill();
        CTX.restore();
    },
    gameCompleted(indices, player) {
        const off = ENGINE.INI.GRIDPIX / 2;
        const CTX = LAYER.strike;
        CTX.lineWidth = 5;
        CTX.strokeStyle = player;

        for (const i of indices) {
            const win = BOARD._ALL_WINDOWS[i];                                          // [[x,y], [x,y], [x,y], [x,y]]
            // first and last cells are valid endpoints for the strike line
            const [x1, y1] = BOARD.gridToCoord(new Grid(win[0][0], win[0][1]), off);
            const [x2, y2] = BOARD.gridToCoord(new Grid(win[3][0], win[3][1]), off);

            CTX.beginPath();
            CTX.moveTo(x1, y1);
            CTX.lineTo(x2, y2);
            CTX.stroke();
        }

        GAME.completed = true;
        this.winner = player;
    },
    getInput(allowed = null) {
        this.awaitingInput = true;
        this.lastInput = null;
        this.allowed = allowed;
    },
};

const GAME = {
    map: null,
    completed: null,
    start() {
        console.log("GAME started");
        if (AUDIO.Title) {
            AUDIO.Title.pause();
            AUDIO.Title.currentTime = 0;
        }
        $(ENGINE.topCanvas).off("mousemove", ENGINE.mouseOver);
        $(ENGINE.topCanvas).off("click", ENGINE.mouseClick);
        $(ENGINE.topCanvas).css("cursor", "");
        ENGINE.hideMouse();
        ENGINE.GAME.pauseBlock();
        ENGINE.GAME.paused = true;

        let GameRD = new RenderData("CompSmooth", 60, "#fF20AA", "text", "#444444", 2, 2, 2);
        ENGINE.TEXT.setRD(GameRD);
        ENGINE.watchVisibility(ENGINE.GAME.lostFocus);
        ENGINE.GAME.setGameLoop(GAME.run);
        ENGINE.GAME.start(16);
        GAME.completed = false;
        GAME.map = new C4Grid(INI.COLS, INI.ROWS + 1);
        GAME.fps = new FPS_short_term_measurement(300);
        GAME.prepareForRestart();
        TURN_MANAGER.init();
        SUBTITLE.init("subtitle", 24, "CompSmooth");
        const nextPlayer = TURN_MANAGER.name[TURN_MANAGER.players[TURN_MANAGER.nextPlayerIndex]];
        SUBTITLE.subtitle(`${nextPlayer}: thinking`, nextPlayer);
        GAME.levelExecute();
    },
    levelExecute() {
        console.info("------------ GAME starts ------------ ");
        if (TURN_MANAGER.runCounter) console.info(`${TURN_MANAGER.runCounter} games remaining`);
        GAME.drawFirstFrame();
        ENGINE.GAME.resume();
    },
    prepareForRestart() {
        let clear = ["background", "text", "FPS", "button", "bottomText", "subtitle", "token", "strike", "grid", "front"];
        ENGINE.clearManylayers(clear);
        TITLE.blackBackgrounds();
        ENGINE.TIMERS.clear();
    },
    setup() {
        console.log("GAME SETUP started");
        for (let agent of Object.keys(AGENT)) {
            for (let player of ["red", "blue"]) {
                $(`#${player}_player_agents`).append(`<option value="${agent}">${agent}</option>`);
            }
        }

        //Default settings:
        $(`#red_player_agents`).val("Human");
        $(`#blue_player_agents`).val("Prophet");
    },
    setTitle() {
        const text = GAME.generateTitleText();
        const RD = new RenderData("CompSmooth", 24, "#0E0", "bottomText");
        const SQ = new RectArea(0, 0, LAYER.bottomText.canvas.width, LAYER.bottomText.canvas.height);
        GAME.movingText = new MovingText(text, 4, RD, SQ);
    },
    generateTitleText() {
        let text = `${PRG.NAME} ${PRG.VERSION
            }, a game by Lovro Selič, ${"\u00A9"} LaughingSkull ${PRG.YEAR
            }. 
             
            Music: 'There's No There There' written and performed by LaughingSkull, ${"\u00A9"
            } 2018 Lovro Selič. `;
        text += "     ENGINE, GRID and GAME code by Lovro Selič using JavaScript. ";
        text = text.split("").join(String.fromCharCode(8202));
        return text;
    },
    runTitle() {
        if (ENGINE.GAME.stopAnimation) return;
        GAME.movingText.process();
        GAME.titleFrameDraw();
    },
    titleFrameDraw() {
        GAME.movingText.draw();
    },
    drawFirstFrame() {
        TITLE.firstFrame();
        BOARD.drawFront();
        BOARD.drawContent();
    },
    run(lapsedTime) {
        if (ENGINE.GAME.stopAnimation) return;
        GAME.respond(lapsedTime);
        TURN_MANAGER.manage(lapsedTime);
        GAME.frameDraw(lapsedTime);
        if (GAME.completed) GAME.complete();
    },
    frameDraw(lapsedTime) {
        if (DEBUG.FPS) {
            GAME.FPS(lapsedTime);
        }
        TURN_MANAGER.drawToken();
    },
    respond(lapsedTime) {
        ENGINE.GAME.respond(lapsedTime);
        const map = ENGINE.GAME.keymap;

        let pressedKeys = ENGINE.GAME.getPressedKeys();
        pressedKeys = pressedKeys.filter(key => TURN_MANAGER.allowed.includes(parseInt(key, 10)));

        if (pressedKeys.length > 0) {
            const key = parseInt(pressedKeys[0], 10);
            TURN_MANAGER.awaitingInput = false;
            TURN_MANAGER.lastInput = key - 1;                                                               //convert key to zero based move
            ENGINE.GAME.keymap[ENGINE.KEY.map[key]] = false;
            return;
        }

        //debug
        if (map[ENGINE.KEY.map.F7]) {
            if (!DEBUG.keys) return;
            throw "Breaking execution!";
        }
        if (map[ENGINE.KEY.map.F8]) {
            if (!DEBUG.keys) return;
            console.log("#####################################");
            console.info("BOARD", GAME.map);
            console.log("#####################################");
            ENGINE.GAME.keymap[ENGINE.KEY.map.F8] = false;
        }

        return;
    },
    FPS(lapsedTime) {
        let CTX = LAYER.FPS;
        CTX.fillStyle = "white";
        ENGINE.clearLayer("FPS");
        let fps = 1000 / lapsedTime || 0;
        GAME.fps.update(fps);
        CTX.fillText(GAME.fps.getFps(), 5, 10);
    },
    gameOverRun(lapsedTime) {
        if (ENGINE.GAME.stopAnimation) return;
        if (ENGINE.GAME.keymap[ENGINE.KEY.map.enter]) {
            ENGINE.GAME.ANIMATION.waitThen(TITLE.startTitle);
        }
        GAME.gameOverFrameDraw(lapsedTime);
    },
    gameOverFrameDraw(lapsedTime) {
        if (DEBUG.FPS) {
            GAME.FPS(lapsedTime);
        }
    },
    complete() {
        const winner = TURN_MANAGER.winner;
        ENGINE.GAME.pauseBlock();
        ENGINE.GAME.ANIMATION.stop();
        const layersToClear = ["FPS"];
        layersToClear.forEach(item => ENGINE.layersToClear.add(item));
        ENGINE.clearLayerStack();

        if (TURN_MANAGER.mode === 1) {
            let subtitleText = `${TURN_MANAGER.name[winner]} wins.`;
            let color = winner;
            if (winner === "Tie") {
                subtitleText = "The game is tied.";
                color = "white";
            }
            subtitleText += " Press <ENTER> for next round using the same settings or <SPACE> to restart.";
            SUBTITLE.subtitle(subtitleText, color);
            ENGINE.GAME.ANIMATION.next(GAME.completeRun);
        } else {
            console.info("game completes in analyze mode", "WINNER:", winner);
            TURN_MANAGER.runCounter--;

            if (winner === "Tie") {
                TURN_MANAGER.ANALYSIS.Tie++
            } else {
                TURN_MANAGER.ANALYSIS[TURN_MANAGER.agent[winner]]++;
            }

            if (TURN_MANAGER.runCounter === 0) return GAME.showAnalysis();
            GAME.start();
        }
    },
    completeRun(lapsedTime) {
        if (ENGINE.GAME.stopAnimation) return;
        if (ENGINE.GAME.keymap[ENGINE.KEY.map.enter]) ENGINE.GAME.ANIMATION.next(GAME.start);
        if (ENGINE.GAME.keymap[ENGINE.KEY.map.space]) ENGINE.GAME.ANIMATION.next(TITLE.startTitle);
    },
    showAnalysis() {
        TURN_MANAGER.analysis_in_progress = false;
        console.clear();
        console.info("*****************************");
        console.info("Analysis results:");
        console.table(TURN_MANAGER.ANALYSIS);
        console.info("Starts:");
        console.table(TURN_MANAGER.STARTS);
        console.info("*****************************");
        TITLE.startTitle();
    }
};

const TITLE = {
    stack: {
    },
    startTitle() {
        if (DEBUG.VERBOSE) console.log("TITLE started");
        if (AUDIO.Title) AUDIO.Title.play();
        ENGINE.GAME.pauseBlock();
        TITLE.clearAllLayers();
        TITLE.blackBackgrounds();
        TITLE.titlePlot();
        ENGINE.draw("background", (ENGINE.gameWIDTH - TEXTURE.Title.width) / 2, (ENGINE.gameHEIGHT - TEXTURE.Title.height) / 2, TEXTURE.Title);
        $("#DOWN")[0].scrollIntoView();
        ENGINE.topCanvas = ENGINE.getCanvasName("ROOM");
        TITLE.drawButtons();
        GAME.setTitle();
        ENGINE.GAME.start(16);
        ENGINE.GAME.ANIMATION.next(GAME.runTitle);
    },
    clearAllLayers() {
        ENGINE.layersToClear = new Set(["text", "sideback", "button", "title", "FPS",
            "bottomText", "subtitle", "token", "strike", "grid", "front", "red", "blue", "col_labels", "strike"]);
        ENGINE.clearLayerStack();
    },
    blackBackgrounds() {
        this.topBackground();
        this.bottomBackground();
        this.sideBackground();
        ENGINE.fillLayer("background", "#000");
    },
    topBackground() {
        const CTX = LAYER.title;
        CTX.fillStyle = "#000";
        CTX.roundRect(0, 0, ENGINE.titleWIDTH, ENGINE.titleHEIGHT, { upperLeft: 20, upperRight: 20, lowerLeft: 0, lowerRight: 0 }, true, true);
    },
    bottomBackground() {
        const CTX = LAYER.bottom;
        CTX.fillStyle = "#000";
        CTX.roundRect(0, 0, ENGINE.bottomWIDTH, ENGINE.bottomHEIGHT, { upperLeft: 0, upperRight: 0, lowerLeft: 20, lowerRight: 20 }, true, true);
    },
    sideBackground() {
        ENGINE.fillLayer("sideback", "#000");
        ENGINE.fillLayer("Lsideback", "#000");
    },
    makeGrad(CTX, x, y, w, h) {
        // Create a linear gradient from (x, y) to (w, h)
        let grad = CTX.createLinearGradient(x, y, w, h);

        grad.addColorStop(0.00, "#0000AA");
        grad.addColorStop(0.05, "#0000BB");
        grad.addColorStop(0.10, "#0000CC");
        grad.addColorStop(0.15, "#0000DD");
        grad.addColorStop(0.20, "#0000EE");
        grad.addColorStop(0.25, "#0000FF");
        grad.addColorStop(0.30, "#8800FF");
        grad.addColorStop(0.35, "#AA00FF");
        grad.addColorStop(0.40, "#CC00EE");
        grad.addColorStop(0.45, "#DD00DD");
        grad.addColorStop(0.50, "#EE00CC");
        grad.addColorStop(0.55, "#FF00AA");
        grad.addColorStop(0.60, "#FF0088");
        grad.addColorStop(0.65, "#FF007F");
        grad.addColorStop(0.70, "#FF0045");
        grad.addColorStop(0.75, "#FF0000");
        grad.addColorStop(0.80, "#EE0000");
        grad.addColorStop(0.85, "#DD0000");
        grad.addColorStop(0.90, "#CC0000");
        grad.addColorStop(0.95, "#BB0000");
        grad.addColorStop(1.00, "#AA0000");

        return grad;
    },
    titlePlot() {
        const CTX = LAYER.title;
        const fs = 64;
        CTX.font = fs + "px CompSmooth";
        CTX.textAlign = "center";
        let txt = CTX.measureText(PRG.NAME);
        let x = ENGINE.titleWIDTH / 2;
        let y = fs;
        let gx = x - txt.width / 2;
        let gy = y - fs;
        let grad = this.makeGrad(CTX, gx, gy + 10, gx, gy + fs);
        CTX.fillStyle = grad;
        GAME.grad = grad;
        CTX.shadowColor = "#666666";
        CTX.shadowOffsetX = 2;
        CTX.shadowOffsetY = 2;
        CTX.shadowBlur = 3;
        CTX.fillText(PRG.NAME, x, y);
    },
    drawButtons() {
        ENGINE.clearLayer("button");
        FORM.BUTTON.POOL.clear();
        let x = 8;
        const w = 100;
        const h = 24;
        const F = 1.5;
        let y = 668;

        const buttonColors = new ColorInfo("#F00", "#A00", "#222", "#666", 13);
        const musicColors = new ColorInfo("#0E0", "#090", "#222", "#666", 13);

        y += F * h;
        let startBA = new Area(x, y, w, h);
        FORM.BUTTON.POOL.push(new Button("Start game", startBA, buttonColors, GAME.start));

        y += F * h;
        let music = new Area(x, y, w, h);
        FORM.BUTTON.POOL.push(new Button("Title music", music, musicColors, TITLE.music));

        FORM.BUTTON.draw();
        $(ENGINE.topCanvas).on("mousemove", { layer: ENGINE.topCanvas }, ENGINE.mouseOver);
        $(ENGINE.topCanvas).on("click", { layer: ENGINE.topCanvas }, ENGINE.mouseClick);
    },
    firstFrame() {
        TITLE.titlePlot();
        ENGINE.clearLayer("bottomText");
        TITLE.score();
    },
    music() {
        AUDIO.Title.play();
    },
    score() {
        const y = ENGINE.gameHEIGHT / 2;
        const X = 16;
        const fs = 15;
        ENGINE.clearLayer("red");
        ENGINE.clearLayer("blue");

        //red
        let CTX = LAYER.red;
        CTX.textAlign = "left";
        CTX.fillStyle = "red";
        CTX.font = `${fs}px CompSmooth`;
        CTX.fillText(`Name: ${TURN_MANAGER.name.red}`, X, y);
        CTX.fillText(`Agent: ${TURN_MANAGER.agent.red}`, X, y + 1.5 * fs);
        CTX.fillText(`Score: ${TURN_MANAGER.score.red}`, X, y + 3 * fs);

        //blue
        CTX = LAYER.blue;
        CTX.textAlign = "left";
        CTX.fillStyle = "blue";
        CTX.font = `${fs}px CompSmooth`;
        CTX.fillText(`Name: ${TURN_MANAGER.name.blue}`, X, y);
        CTX.fillText(`Agent: ${TURN_MANAGER.agent.blue}`, X, y + 1.5 * fs);
        CTX.fillText(`Score: ${TURN_MANAGER.score.blue}`, X, y + 3 * fs);
    },
};

// -- main --
$(function () {
    PRG.INIT();
    PRG.setup();
    ENGINE.LOAD.preload();
});