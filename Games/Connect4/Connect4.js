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
    - GRID : REVERSE DICT ENGINE  VERBOSE bug corrected;;


 */
////////////////////////////////////////////////////

const DEBUG = {
    SETTING: true,
    AUTO_TEST: false,
    FPS: true,
    VERBOSE: true,
    max17: false,
    keys: true,
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
};

const PRG = {
    VERSION: "0.2.5",
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
        ENGINE.addBOX("ROOM", ENGINE.gameWIDTH, ENGINE.gameHEIGHT, ["background", "board", "front", "grid", "col_labels", "text", "FPS", "button", "click"], "side");
        ENGINE.addBOX("SIDE", ENGINE.sideWIDTH, ENGINE.gameHEIGHT, ["sideback", "blue",], "fside");
        ENGINE.addBOX("DOWN", ENGINE.bottomWIDTH, ENGINE.bottomHEIGHT, ["bottom", "bottomText", "subtitle"], null);

        /* Connect-4 overrides */
        ENGINE.setGridSize(INI.GRIDSIZE);
        MAPDICT.RED = 1;
        MAPDICT.BLUE = 2;

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
            const y = GS * 0.75;
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
    gridToCoord(grid) {
        const GS = ENGINE.INI.GRIDPIX;
        let x = grid.x * GS + INI.LEFT_X;
        let y = ENGINE.gameHEIGHT - (grid.y + 1) * GS;
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
};

const AGENT = {
    Human() {
        console.info("*** Human ***");
        console.time("human");

        console.timeEnd("human");
        console.info("*************\n");
    },
    Random() {
        console.info("*** Random ***");
        console.time("random");
        let legal_moves = AGENT_MANAGER.getLegalMoves();
        console.log(".legal_moves", legal_moves);

        console.timeEnd("random");
        console.info("*************\n");
    }
};

const AGENT_MANAGER = {
    //map --> GAME.map
    getLegalMoves() {
        const legalMoves = [];
        for (let c = 0; c < INI.COLS; c++) {
            let grid = new Grid(c, INI.ROWS - 1);
            if (GAME.map.isZero(grid)) legalMoves.push(c);
        }
        return legalMoves;
    }
};

class Token {
    constructor(move) {
        this.move = move;

    }
}

const TURN_MANAGER = {
    nextPlayerIndex: null,
    players: ["red", "blue"],
    turn_completed: null,
    turn: null,
    agent: {
        red: null,
        blue: null
    },
    mode: 1,                        //game (default)
    token: null,
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
        if ($('input[name="mode"]:checked').val() === "analyze") this.mode = 0;
        this.turn = 0;
        this.turn_completed = true;
        this.agent.red = $("#red_player_agents")[0].value;
        this.agent.blue = $("#blue_player_agents")[0].value;
        /////////////
        console.info("Agents:");
        console.table(TURN_MANAGER.agent);
        console.info("Mode:", this.mode);
    },
    getPlayer() {
        let player = this.players[this.nextPlayerIndex];
        this.nextPlayerIndex++;
        this.nextPlayerIndex %= 2;
        return player;
    },
    nextPlayer() {
        this.turn++;
        if (this.turn === INI.OVER_TURN) {
            //game is tied
            throw `Tied game from overturn on turn ${this.turn}!`;
        }

        let player = this.getPlayer();
        console.log(`Turn ${this.turn}, player: ${player}, agent: ${this.agent[player]}`);
        const move = AGENT[this.agent[player]]();
        console.log(".move", move);
        if (this.mode) {
            this.setMove(move);
        } else this.applyMove(move);

        //calculate and draw score
        //check if player has won
        console.log("-------------------------------\n");
    },
    setMove(move) {
        console.log(".. setting move", move);
    },
    applyMove(move) {
        console.log(".. applying move", move);
    },
    animateMove() {
        console.log("... animating move", this.token);
    },
    manage() {
        if (this.turn_completed) this.nextPlayer();
    }
};

const GAME = {
    map: null,
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

        let GameRD = new RenderData("CompSmooth", 60, "#fF2010", "text", "#444444", 2, 2, 2);
        ENGINE.TEXT.setRD(GameRD);
        ENGINE.watchVisibility(ENGINE.GAME.lostFocus);
        ENGINE.GAME.setGameLoop(GAME.run);
        ENGINE.GAME.start(16);
        GAME.completed = false;
        GAME.map = new C4Grid(INI.COLS, INI.ROWS + 1);
        GAME.fps = new FPS_short_term_measurement(300);
        GAME.prepareForRestart();
        TURN_MANAGER.init();
        //BOARD.debugBoard();
        GAME.levelExecute();
    },
    levelExecute() {
        console.error("GAME starts");
        GAME.drawFirstFrame();
        ENGINE.GAME.ANIMATION.next(GAME.run);
    },
    prepareForRestart() {
        let clear = ["background", "text", "FPS", "button", "bottomText", "subtitle"];
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
        //defaults to random
        for (let player of ["red", "blue"]) {
            $(`#${player}_player_agents`).val("Random");
        }
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
        TURN_MANAGER.manage();
        if (TURN_MANAGER.token) TURN_MANAGER.animateMove();
        //ENGINE.TIMERS.update();

        GAME.frameDraw(lapsedTime);
        //if (GAME.completed) GAME.won();
    },
    frameDraw(lapsedTime) {
        if (DEBUG.FPS) {
            GAME.FPS(lapsedTime);
        }
    },
    respond(lapsedTime) {
        //if (HERO.dead) return;

        //HERO.player.respond(lapsedTime);
        //WebGL.GAME.respond(lapsedTime);
        ENGINE.GAME.respond(lapsedTime);

        const map = ENGINE.GAME.keymap;

        //debug
        if (map[ENGINE.KEY.map.F7]) {
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

        //controls


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
    won() {
        console.info("GAME WON");
        TITLE.setEndingCreditsScroll();
        ENGINE.GAME.pauseBlock();
        const layersToClear = ["FPS", "info"];
        layersToClear.forEach(item => ENGINE.layersToClear.add(item));
        ENGINE.clearLayerStack();
        ENGINE.GAME.ANIMATION.stop();
        const delay = 4000;
        setTimeout(function () {
            ENGINE.clearLayer("subtitle");
            TITLE.music();
            ENGINE.GAME.ANIMATION.next(GAME.wonRun);
        }, delay);
    },
    wonRun(lapsedTime) {
        if (ENGINE.GAME.stopAnimation) return;
        GAME.endingCreditText.process(lapsedTime);
        GAME.wonFrameDraw();
        if (ENGINE.GAME.keymap[ENGINE.KEY.map.enter]) {
            ENGINE.GAME.ANIMATION.next(TITLE.startTitle);
        }
    },
    wonFrameDraw() {
        GAME.endingCreditText.draw();
    },
};

const TITLE = {
    scoreY: null,
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
        ENGINE.layersToClear = new Set(["text", "sideback", "button", "title", "FPS", "bottomText", "subtitle"]);
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
        TITLE.side();
    },
    music() {
        AUDIO.Title.play();
    },
    side() {
        const y = ENGINE.gameHEIGHT / 2;
        const X = 16;
        const fs = 15;

        //red
        let CTX = LAYER.red;
        CTX.textAlign = "left";
        CTX.fillStyle = "red";
        CTX.font = `${fs}px CompSmooth`;
        CTX.fillText(`Name: ${$("#red_player_name")[0].value}`, X, y);
        CTX.fillText(`Agent: ${$("#red_player_agents")[0].value}`, X, y + 1.5 * fs);

        //blue
        CTX = LAYER.blue;
        CTX.textAlign = "left";
        CTX.fillStyle = "blue";
        CTX.font = `${fs}px CompSmooth`;
        CTX.fillText(`Name: ${$("#blue_player_name")[0].value}`, X, y);
        CTX.fillText(`Agent: ${$("#blue_player_agents")[0].value}`, X, y + 1.5 * fs);

        //stack coords
        this.scoreY = y + 4 + fs;
    }
};

// -- main --
$(function () {
    SPEECH.init();
    PRG.INIT();
    PRG.setup();
    ENGINE.LOAD.preload();
});