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
    AUTO_TEST: false,
    FPS: true,
    VERBOSE: true,
    max17: false,
    keys: true,
    simulation: false,
    board: [
        0, 2, 1, 2, 2, 2, 0,
        0, 1, 1, 1, 2, 0, 0,
        0, 2, 1, 2, 0, 0, 0,
        0, 1, 2, 2, 0, 0, 0,
        0, 0, 1, 1, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0,
    ],
    test() {
        console.warn("-------------------------------------------------");
        let [patterns, coordinates] = BOARD.boardToPatterns([2]);
        console.log(patterns);
        console.log(coordinates);
        /*
        console.warn("-------------------------------------------------");
        let [patterns2, coordinates2] = BOARD.boardToPatterns([2]);
        console.log(patterns2);
        console.log(coordinates2);
        console.warn("-------------------------------------------------");
        let [patterns3, coordinates3] = BOARD.boardToPatterns([1, 2]);
        console.log(patterns3);
        console.log(coordinates3);
        */
        console.warn("-------------------------------------------------");
        //let check_2_1 = BOARD.countWindowsInPattern(patterns1, 2, 1);
        //console.log("check_2_1", check_2_1);
        let check_4_2 = BOARD.countWindowsInPattern(patterns, 4, 2);
        console.log("check_4_2", check_4_2);

    },
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
    INROW3: 64,
    INROW4: 1000,
};

const PRG = {
    VERSION: "0.5.7",
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
            //$("#maze_version").html(DUNGEON.VERSION);
            //$("#iam_version").html(IndexArrayManagers.VERSION);
            $("#lib_version").html(LIB.VERSION);
            //$("#webgl_version").html(WebGL.VERSION);
            //$("#maptools_version").html(MAP_TOOLS.VERSION);
            $("#speech_version").html(SPEECH.VERSION);
        } else {
            $('#debug').hide();
        }

        $("#toggleHelp").click(function () {
            $("#help").toggle(400);
        });
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
        ENGINE.addBOX("ROOM", ENGINE.gameWIDTH, ENGINE.gameHEIGHT, ["background", "token", "grid", "front", "strike", "col_labels", "text", "FPS", "button", "click"], "side");
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

            console.warn(" *** verbose setting ***");
            console.warn("ENGINE:", ENGINE.verbose);
        }
    },
    start() {
        console.log("%c**************************************************************************************************************************************", PRG.CSS);
        console.log(`${PRG.NAME} ${PRG.VERSION} STARTED!`);
        console.log("%c**************************************************************************************************************************************", PRG.CSS);
        $(ENGINE.topCanvas).off("mousemove", ENGINE.mouseOver);
        $(ENGINE.topCanvas).off("click", ENGINE.mouseClick);
        $(ENGINE.topCanvas).css("cursor", "");

        if (SPEECH.VERBOSE) {
            console.info("SPEECH available voices");
            console.table(SPEECH.voices);
        }

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
        this.drawTopGrid(CTX, GS);
        this.drawColLabels(CTX, GS);
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
    drawColLabels(CTX, GS) {
        const GS2 = Math.floor(GS / 2);
        const fs = 42;
        CTX.font = `${fs}px CompSmooth`;
        CTX.textAlign = "center";
        CTX.fillStyle = "rgba(100, 100, 100, 0.3)";
        for (let x = 0; x < INI.COLS; x++) {
            const y = GS * 0.5;
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
    debugBoard() {
        //reds
        GAME.map.toRed(new Grid(0, 0));
        GAME.map.toRed(new Grid(0, INI.ROWS - 1));
        //blues
        GAME.map.toBlue(new Grid(INI.COLS - 1, 0));
        GAME.map.toBlue(new Grid(INI.COLS - 1, INI.ROWS - 1));
        //red top row
        GAME.map.toRed(new Grid(0, INI.ROWS));
        //
        console.info("debugBoard", GAME.map);
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
        const uniquePlayers = new Set(pattern.filter(v => v !== 0)).size;
        const playerWith2OrMore = players.some(p => pattern.filter(v => v === p).length >= 2);


        // !(zeros === 0 && uniquePlayers > 1) && 
        return (
            zeros < 3 &&
            matches >= 2 &&
            //!(zeros === 0 && uniquePlayers > 1) &&
            playerWith2OrMore
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
    }

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
    Village_Idiot(playerIndex) {
        console.info("*** Village_Idiot ***");
        console.time("Village_Idiot");
        let move = AGENT_MANAGER.N_step_lookahead(playerIndex, 2);
        console.timeEnd("Village_Idiot");
        console.info("*************\n");
        return move;
    },
    Friendly(playerIndex) {
        console.info("*** Friendly ***");
        console.time("Friendly");
        let move = AGENT_MANAGER.N_step_lookahead(playerIndex, 3);
        console.timeEnd("Friendly");
        console.info("*************\n");
        return move;
    },
    Smarty(playerIndex) {
        console.info("*** Smarty ***");
        console.time("Smarty");
        let move = AGENT_MANAGER.N_step_lookahead(playerIndex, 4);
        console.timeEnd("Smarty");
        console.info("*************\n");
        return move;
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
    },
    getDestination(move) {
        return this.getEmptyRow(GAME.map, move);
    },
    N_step_lookahead(playerIndex, N) {
        console.info("### N_step_lookahead ###", "playerIndex", playerIndex, "N", N);
        let moves = this.getLegalCentreOrderedMoves();
        const scores = {};
        for (const move of moves) {
            scores[move] = this.scoreMove(GAME.map, move, playerIndex, N);
        }
        console.table(scores);
        const maxScore = Math.max(...Object.values(scores));
        const bestMoves = Object.entries(scores)
            .filter(([_, score]) => score === maxScore)
            .map(([move]) => parseInt(move));

        const innermost = this.innermost(bestMoves);
        console.log("best moves", bestMoves, "innermost", innermost);

        return innermost;
    },
    scoreMove(grid, move, playerIndex, N) {
        //console.group("MOVE");
        //console.warn("\n-------------------------------\nscoring move", move, "playerIndex", playerIndex, "N", N);
        let nextGrid_GA = this.dropPiece(grid, move, playerIndex);                                                          //GA! - cloned
        const patterns = BOARD.boardToPatterns([1, 2], nextGrid_GA)[0];
        let score = this.minimax(nextGrid_GA, N - 1, false, playerIndex, -Infinity, Infinity, patterns);
        //console.log("... scoreMove", "move", move, "score", score);
        //console.groupEnd("MOVE");

        return score;
    },
    dropPiece(grid, move, playerIndex) {
        let nextGrid = grid.clone();                                                                                        //this is GA!
        let placedGrid = this.getEmptyRow(nextGrid, move);                                                                  //filtered for valid moves
        nextGrid.setValue(placedGrid, playerIndex);
        //console.info(move, "player", TURN_MANAGER.players[playerIndex - 1]);
        BOARD.printBoardToConsole(nextGrid);
        return nextGrid;
    },
    minimax(GA, depth, maximizingPlayer, playerIndex, A, B, patterns) {
        //console.error("..minimax depth", depth, "maximizingPlayer", maximizingPlayer, "player", TURN_MANAGER.players[playerIndex - 1]);
        if (depth === 0 || this.isTerminalNode(GA, patterns)) return this.getHeuristic(playerIndex, patterns);
        const validMoves = this.getLegalCentreOrderedMoves(GA);

        if (maximizingPlayer) {
            let value = -Infinity;
            for (const col of validMoves) {
                const childGA = this.dropPiece(GA, col, playerIndex);
                const childPatterns = BOARD.boardToPatterns([1, 2], childGA)[0];
                const newValue = this.minimax(childGA, depth - 1, false, playerIndex, A, B, childPatterns);
                value = Math.max(value, newValue);
                if (value >= B) break;
                A = Math.max(A, value);
            }
            return value;
        } else {
            let value = Infinity;
            const opponent = playerIndex % 2 + 1;
            for (const col of validMoves) {
                const childGA = this.dropPiece(GA, col, opponent);
                const childPatterns = BOARD.boardToPatterns([1, 2], childGA)[0];
                const newValue = this.minimax(childGA, depth - 1, true, playerIndex, A, B, childPatterns);
                value = Math.min(value, newValue);
                if (value <= A) break;
                B = Math.min(B, value);
            }
            return value;
        }
    },
    getHeuristic(playerIndex, patterns) {
        const pieces = [2, 3, 4];
        const oppoPlayer = playerIndex % 2 + 1;
        const player = pieces.map(p => BOARD.countWindowsInPattern(patterns, p, playerIndex));
        const oppo = pieces.map(p => BOARD.countWindowsInPattern(patterns, p, oppoPlayer));
        const score = pieces.reduce((sum, n, i) => {
            const weight = INI[`INROW${n}`];
            return sum + weight * (player[i].count - oppo[i].count);
        }, 0);

        //console.warn("....score after terminal or depth", score);
        //console.log("\n");
        return score;
    },
    isTerminalNode(GA, patterns) {
        //Check for draw - this is redundant?
        let topRow = Array.from(GA.map.slice(-INI.COLS));                                           //GA.map is UInt8Array
        //console.log("..isTerminalNode; topRow", topRow);
        //if (topRow.count(0) === 0) return true;
        if (topRow.count(0) === 0) {
            console.error("topRow full");
            throw "topRow full";
            return true;
        }

        //check for win
        const FourInARow = BOARD.countWindowsInPattern(patterns, 4, 1).count + BOARD.countWindowsInPattern(patterns, 4, 2).count;
        return FourInARow > 0;
    },
    innermost(arr) {
        const mid = (INI.COLS - 1) / 2;
        let bestMove = arr[0];
        let bestScore = -Math.abs(arr[0] - mid);

        for (let i = 1; i < arr.length; i++) {
            const score = -Math.abs(arr[i] - mid);
            if (score > bestScore) {
                bestScore = score;
                bestMove = arr[i];
            }
        }

        return bestMove;
    }
}

class Token {
    constructor(move, startGrid, destination, player) {
        this.move = move;
        this.moveState = new MoveState(startGrid, UP, GAME.map, FP_Grid.toClass(startGrid));
        this.destination = destination;
        this.player = player;
        this.onDestination = false;
        //this.moveSpeed = 6;
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
    init() {
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
        //this.nextPlayerIndex = 0; //debug red starts
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
        /////////////
        console.info("Agents:");
        console.table(TURN_MANAGER.agent);
        console.info("Mode:", this.mode);
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
        console.clear();
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
                //console.error(`Tied game from overturn on turn ${this.turn}!`);
                return;
            }

            player = this.getPlayer();
            //console.warn(this.agent[player]);
            SUBTITLE.subtitle(`${this.name[player]}: thinking`, player);
            move = AGENT[this.agent[player]](this.playerToIndex(player) + 1);


            //console.log(`\n\nTurn ${this.turn}, player: ${player}, agent: ${this.agent[player]}, move: ${move}`);
        }

        if (TURN_MANAGER.awaitingInput) {
            SUBTITLE.subtitle(`${this.name[player]}: waiting for input`, player);
            return;
        }


        this.turn_completed = false;
        const destination = AGENT_MANAGER.getDestination(move);
        //console.info("DESTINATION", destination, GAME.map.getValue(destination));

        if (this.mode) {
            this.setMove(move, destination, player);
        } else this.applyDestination(destination);

        SUBTITLE.subtitle(`${this.name[player]}: column ${move + 1}`, player);
    },
    setMove(move, destination, player) {
        this.token = new Token(move, new Grid(move, INI.ROWS), destination, player);
    },
    applyDestination(destination, player) {
        this.turn_completed = true;
        this.token = null;
        GAME.map.setToken(destination, player);
        //console.info("applyDestination", destination, "player", player);
        BOARD.drawContent();

        //analyze
        [BOARD.patterns, BOARD.coordinates] = BOARD.boardToPatterns([this.playerPieces[player]]);

        //check if player has won
        const winCheck = BOARD.countWindowsInPattern(BOARD.patterns, 4, this.playerPieces[player]);
        const win = winCheck.count > 0;
        if (win) return this.gameCompleted(winCheck.indices, player);

        //calculate and draw score
        const inrow2 = BOARD.countWindowsInPattern(BOARD.patterns, 2, this.playerPieces[player]).count;
        const inrow3 = BOARD.countWindowsInPattern(BOARD.patterns, 3, this.playerPieces[player]).count;
        TURN_MANAGER.score[player] = inrow2 * INI.INROW2 + inrow3 * INI.INROW3;
        TITLE.score();

        //DEBUG
        //if (this.turn === 4) throw "debug after turn 4";
        //
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
        console.warn("---- game completed ----", indices, player);
        const off = ENGINE.INI.GRIDPIX / 2;
        const CTX = LAYER.strike;
        CTX.lineWidth = 5;
        CTX.strokeStyle = player;
        for (const i of indices) {
            console.info("....", i, "coords", BOARD.coordinates[i]);
            let [x1, y1] = BOARD.gridToCoord(new Grid(BOARD.coordinates[i][0][0], BOARD.coordinates[i][0][1]), off);
            let [x2, y2] = BOARD.gridToCoord(new Grid(BOARD.coordinates[i][3][0], BOARD.coordinates[i][3][1]), off);

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
        //BOARD.debugBoard();
        //BOARD.importBoard(DEBUG.board);
        GAME.levelExecute();
    },
    levelExecute() {
        console.info("------------ GAME starts ------------ ");
        GAME.drawFirstFrame();


        //DEBUG.test();




        //ENGINE.GAME.ANIMATION.next(GAME.run);

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

        //$(`#red_player_agents`).val("Human");
        //$(`#red_player_agents`).val("Random");
        $(`#red_player_agents`).val("Smarty");
        //$(`#red_player_agents`).val("Friendly");
        //$(`#red_player_agents`).val("Village_Idiot");
        //$(`#blue_player_agents`).val("Random");
        //$(`#blue_player_agents`).val("Human");
        //$(`#blue_player_agents`).val("Human");
        $(`#blue_player_agents`).val("Friendly");
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
        text += "     ENGINE, SPEECH, GRID and GAME code by Lovro Selič using JavaScript. ";
        text = text.split("").join(String.fromCharCode(8202));
        return text;
    },
    runTitle() {
        if (ENGINE.GAME.stopAnimation) return;
        GAME.movingText.process();
        GAME.titleFrameDraw();
        SPEECH.silence();
    },
    titleFrameDraw() {
        GAME.movingText.draw();
    },
    drawFirstFrame() {
        if (DEBUG.VERBOSE) console.log("drawing first frame");
        TITLE.firstFrame();
        BOARD.drawFront();
        BOARD.drawContent();
    },
    run(lapsedTime) {
        if (ENGINE.GAME.stopAnimation) return;
        //const date = Date.now();
        GAME.respond(lapsedTime);

        TURN_MANAGER.manage(lapsedTime);

        //ENGINE.TIMERS.update();

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
            throw "Breaking execution!";
            if (!DEBUG.keys) return;
        }
        if (map[ENGINE.KEY.map.F8]) {
            if (!DEBUG.keys) return;
            console.log("#####################################");
            console.info("BOARD", GAME.map);
            console.log("#####################################");
            ENGINE.GAME.keymap[ENGINE.KEY.map.F8] = false;
        }
        if (map[ENGINE.KEY.map.F9]) {
            if (!DEBUG.keys) return;
            console.log("\nDEBUG:");
            console.log("#######################################################");
            console.log("#######################################################");
            ENGINE.GAME.keymap[ENGINE.KEY.map.F9] = false;
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
        //const date = Date.now();
        GAME.gameOverFrameDraw(lapsedTime);
    },
    gameOverFrameDraw(lapsedTime) {
        if (DEBUG.FPS) {
            GAME.FPS(lapsedTime);
        }
    },
    complete() {
        console.warn("\n---------------------- completing game -------------------------------------------------");

        const winner = TURN_MANAGER.winner;
        console.info("WINNER:", winner);
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
            console.info("game completes in analyze mode");
            //to be implemented
        }

        console.warn("-------------------------------------------------\n");
    },
    completeRun(lapsedTime) {
        if (ENGINE.GAME.stopAnimation) return;
        if (ENGINE.GAME.keymap[ENGINE.KEY.map.enter]) ENGINE.GAME.ANIMATION.next(GAME.start);
        if (ENGINE.GAME.keymap[ENGINE.KEY.map.space]) ENGINE.GAME.ANIMATION.next(TITLE.startTitle);
    },
};

const TITLE = {
    //scoreY: null,
    stack: {
    },
    startTitle() {
        if (DEBUG.VERBOSE) console.log("TITLE started");
        //if (AUDIO.Title) AUDIO.Title.play(); //dev
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
            "bottomText", "subtitle", "token", "strike", "grid", "front", "red", "blue"]);
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
        //CTX.fillText(`Name: ${$("#red_player_name")[0].value}`, X, y);
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
    SPEECH.init();
    PRG.INIT();
    PRG.setup();
    ENGINE.LOAD.preload();
});