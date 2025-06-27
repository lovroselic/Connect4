/*jshint browser: true */
/*jshint -W097 */
/*jshint -W117 */
/*jshint -W061 */
"use strict";

//////////////////////////////////////
// GRID              by LS          //
//////////////////////////////////////

/*
TODO:
    - updated in Connect4, copy to HTH! use same version
  
known bugs:

*/

const GRID = {
    VERSION: "4.00",
    CSS: "color: #0AA",
    SETTING: {
        ALLOW_CROSS: false,
        EPSILON: 0.05,
        FORWARD_CIRCLE_RESOLUTION: 2,
        FORWARD_CIRCLE_CHECK_ANGLE: Math.PI / 4,
    },
    circleCollision(entity1, entity2) {
        let distance = entity1.moveState.pos.EuclidianDistance(entity2.moveState.pos);
        let touchDistance = entity1.r + entity2.r;
        return distance < touchDistance;
    },
    circleCollision2D(fpgrid1, fpgrid2, touchDistance) {
        return fpgrid1.EuclidianDistance(fpgrid2) < touchDistance;
    },
    /**
     * @param {Vector3} pos1vec3
     * @param {Vector3} pos2vec3
     * @param {float} touchDistance
     * @return {bool} 
     */
    circleCollision3D(pos1vec3, pos2vec3, touchDistance) {
        return pos1vec3.EuclidianDistance(pos2vec3) < touchDistance;
    },
    circleCollision_toPos(entity1, entity2) {
        console.log(entity1.moveState.pos, entity2.pos);
        let distance = entity1.moveState.pos.EuclidianDistance(entity2.pos);
        let touchDistance = entity1.r + entity2.r;
        console.log(distance, touchDistance);
        return distance < touchDistance;
    },
    circleRectangleCollision() { },
    collision(actor, grid) {
        let actorGrid = actor.moveState.homeGrid;
        return GRID.same(actorGrid, grid);
    },
    spriteToSpriteCollision(actor1, actor2) {
        return GRID.same(actor1.moveState.homeGrid, actor2.moveState.homeGrid);
    },
    gridToCenterPX(grid) {
        const half = ENGINE.INI.GRIDPIX >>> 1;
        let x = grid.x * ENGINE.INI.GRIDPIX + half;
        let y = grid.y * ENGINE.INI.GRIDPIX + half;
        return new Point(x, y);
    },
    gridToSprite(grid, actor) {
        GRID.coordToSprite(GRID.gridToCoord(grid), actor);
    },
    gridToSpriteBottomCenter(grid, actor) {
        GRID.coordToSpriteBottomCenter(GRID.gridToCoord(grid), actor);
    },
    coordToSprite(coord, actor) {
        const half = ENGINE.INI.GRIDPIX >>> 1;
        actor.x = coord.x + half;
        actor.y = coord.y + half;
    },
    coordToSpriteBottomCenter(coord, actor) {
        const half = ENGINE.INI.GRIDPIX >>> 1;
        actor.x = coord.x + half;
        actor.y = coord.y + ENGINE.INI.GRIDPIX;
    },
    gridToBottomCenterPX(grid) {
        const half = ENGINE.INI.GRIDPIX >>> 1;
        let x = grid.x * ENGINE.INI.GRIDPIX + half;
        let y = (grid.y + 1) * ENGINE.INI.GRIDPIX;
        return new Point(x, y);
    },
    gridToCoord(grid) {
        let x = grid.x * ENGINE.INI.GRIDPIX;
        let y = grid.y * ENGINE.INI.GRIDPIX;
        return new Point(x, y);
    },
    coordToGrid(x, y) {
        const tx = Math.floor(x / ENGINE.INI.GRIDPIX);
        const ty = Math.floor(y / ENGINE.INI.GRIDPIX);
        return new Grid(tx, ty);
    },
    coordToFP_Grid(x, y) {
        var tx = x / ENGINE.INI.GRIDPIX;
        var ty = y / ENGINE.INI.GRIDPIX;
        return new FP_Grid(tx, ty);
    },
    pointToGrid(point) {
        return this.coordToGrid(point.x, point.y);
    },
    grid(CTX = LAYER.grid) {
        var x = 0;
        var y = 0;
        CTX.strokeStyle = "#AAA";
        //horizonal lines
        do {
            y += ENGINE.INI.GRIDPIX;
            CTX.beginPath();
            CTX.setLineDash([1, 3]);
            CTX.moveTo(x, y);
            CTX.lineTo(CTX.canvas.width, y);
            CTX.closePath();
            CTX.stroke();
        } while (y <= CTX.canvas.height);
        //vertical lines
        y = 0;
        do {
            x += ENGINE.INI.GRIDPIX;
            CTX.beginPath();
            CTX.setLineDash([1, 3]);
            CTX.moveTo(x, y);
            CTX.lineTo(x, CTX.canvas.height);
            CTX.closePath();
            CTX.stroke();
        } while (x <= CTX.canvas.width);
    },
    paintCoord(layer, dungeon, all = false) {
        ENGINE.clearLayer(layer);
        for (let x = 0; x < dungeon.width; x++) {
            for (let y = 0; y < dungeon.height; y++) {
                let grid = new Grid(x, y);
                if (all || !dungeon.GA.check(grid, MAPDICT.WALL)) {
                    let point = GRID.gridToCoord(grid);
                    let text = `${x},${y}`;
                    GRID.paintText(point, text, layer, "#BBB");
                    let index = x + dungeon.width * y;
                    point = point.add(DOWN, 12);
                    GRID.paintText(point, index, layer, "#BBB");
                }
            }
        }
    },
    paintCoord3D(layer, dungeon, floor = 0, all = false) {
        ENGINE.clearLayer(layer);
        for (let x = 0; x < dungeon.width; x++) {
            for (let y = 0; y < dungeon.height; y++) {
                let grid = new Grid3D(x, y, floor);
                if (all || !dungeon.GA.check(grid, MAPDICT.WALL)) {
                    let point = GRID.gridToCoord(grid);
                    let text = `${x},${y}`;
                    GRID.paintText(point, text, layer, "#BBB");
                    let index = x + dungeon.width * y + floor * (x + dungeon.width * y);
                    point = point.add(DOWN, 12);
                    GRID.paintText(point, index, layer, "#BBB");
                }
            }
        }
    },
    paintText(point, text, layer, color = "#FFF") {
        const CTX = LAYER[layer];
        CTX.font = "10px Consolas";
        const half = ENGINE.INI.GRIDPIX >>> 1;
        let y = point.y + half;
        let x = point.x + half;
        CTX.fillStyle = color;
        CTX.textAlign = "center";
        CTX.fillText(text, x, y);
    },
    trueToGrid(actor) {
        const GRIDPIX = ENGINE.INI.GRIDPIX;
        const TX = actor.x - (GRIDPIX >> 1);
        const TY = actor.y - (GRIDPIX >> 1);
        const [GX, GY] = [TX / GRIDPIX | 0, TY / GRIDPIX | 0];
        const [MX, MY] = [TX % GRIDPIX, TY % GRIDPIX];
        if (MX || MY) return null;
        return { x: GX, y: GY };
    },
    same(grid1, grid2) {
        return (grid1 && grid2 && grid1.x === grid2.x && grid1.y === grid2.y);
    },
    same3D(grid1, grid2) {
        return (grid1 && grid2 && grid1.x === grid2.x && grid1.y === grid2.y && grid1.z === grid2.z);
    },
    isGridIn(grid, gridArray) {
        return gridArray.findIndex(g => g.x === grid.x && g.y === grid.y);
    },
    contTranslatePosition(entity, lapsedTime) {
        let length = (lapsedTime / 1000) * entity.moveSpeed;
        entity.moveState.pos = entity.moveState.pos.translate(entity.moveState.dir, length);
        entity.actor.updateAnimation(lapsedTime);
        return;
    },
    translatePosition3D(entity, lapsedTime) {
        //console.warn("translatePosition3D", entity);
        const length = (lapsedTime / 1000) * entity.moveSpeed;
        const realDir = Vector3.from_3D_dir(entity.moveState.realDir); //3D to 3D , swap yz
        entity.moveState.pos = entity.moveState.pos.translate(realDir, length);

        const overallDistance = Vector3.to_FP_Grid3D(entity.moveState.pos).EuclidianDistance(entity.moveState.startPos);
        //console.warn(`${entity.name} ${entity.id} overallDistance: ${overallDistance}`);
        if (overallDistance > 1.0) {
            entity.moveState.moving = false;
            //console.info(`${entity.name} ${entity.id} lapsed time: ${lapsedTime} went too far.`);
        }
    },
    translatePosition(entity, lapsedTime) {
        let length = (lapsedTime / 1000) * entity.moveSpeed;
        entity.moveState.pos = entity.moveState.pos.translate(entity.moveState.realDir, length);
        let distance = entity.moveState.pos.EuclidianDistance(entity.moveState.endPos);
        let boundGrid = Grid.toClass(entity.moveState.pos);
        if (!(GRID.same(boundGrid, Grid.toClass(entity.moveState.endPos)) || GRID.same(boundGrid, Grid.toClass(entity.moveState.startPos)))) {
            entity.moveState.pos = entity.moveState.endPos;
            entity.moveState.moving = false;
            return;
        }

        entity.actor.updateAnimation(lapsedTime);

        if (distance < GRID.SETTING.EPSILON) {
            entity.moveState.moving = false;
            return;
        }
    },
    translateMove(entity, lapsedTime, gridArray, changeView = false, onFinish = null, animate = true) {
        if (!gridArray) gridArray = entity.moveState.gridArray;
        entity.actor.x += entity.moveState.dir.x * entity.speed;
        entity.actor.y += entity.moveState.dir.y * entity.speed;
        entity.actor.orientation = entity.actor.getOrientation(entity.moveState.dir);
        if (animate) entity.actor.updateAnimation(lapsedTime, entity.actor.orientation);
        entity.moveState.homeGrid = GRID.coordToGrid(entity.actor.x, entity.actor.y);
        entity.moveState.pos = entity.moveState.homeGrid;
        if (gridArray.outside(entity.moveState.homeGrid)) {
            entity.moveState.homeGrid = gridArray.toOtherSide(entity.moveState.homeGrid);
            GRID.gridToSprite(entity.moveState.homeGrid, entity.actor);
        }
        if (changeView) ENGINE.VIEWPORT.check(entity.actor);
        ENGINE.VIEWPORT.alignTo(entity.actor);
        if (GRID.same(entity.moveState.endGrid, GRID.trueToGrid(entity.actor))) {
            entity.moveState.moving = false;
            entity.moveState.startGrid = entity.moveState.endGrid;
            entity.moveState.homeGrid = entity.moveState.endGrid;
            if (onFinish) onFinish.call();
        }
        return;
    },

    /**
     * does not use actor, only moveState.pos (FP_Grid)
     * no animation, no viewport
     * revertX = 1, revertY = 1; change to -1 to revert axis
     */
    translateMoveState(entity, lapsedTime, revertX = 1, revertY = 1, onFinish = null) {
        let length = (lapsedTime / 1000) * entity.moveSpeed;
        entity.moveState.pos.x += entity.moveState.dir.x * length;
        entity.moveState.pos.y += entity.moveState.dir.y * length;
        entity.moveState.homeGrid = new Grid(
            entity.moveState.pos.x + 0.975 * entity.moveState.dir.x * revertX,
            entity.moveState.pos.y + 0.975 * entity.moveState.dir.y * revertY
        );
        //console.log(".... translateMoveState", "entity.moveState.startGrid.y", entity.moveState.startGrid.y, "entity.moveState.pos.y", entity.moveState.pos.y, "entity.moveState.homeGrid.y", entity.moveState.homeGrid.y);
        if (GRID.same(entity.moveState.endGrid, entity.moveState.homeGrid)) {
            //console.warn("...... grid reached", entity.moveState.endGrid, entity.moveState.homeGrid);
            //console.warn("........... grid Y reached", "entity.moveState.pos.y", entity.moveState.pos.y, "entity.moveState.homeGrid.y", entity.moveState.homeGrid.y);
            entity.moveState.moving = false;
            entity.moveState.startGrid = entity.moveState.endGrid;
            if (onFinish) onFinish.call();
        }
        return;
    },
    blockMove(entity, changeView = false) {
        let newGrid = entity.moveState.startGrid.add(entity.moveState.dir);
        entity.moveState.reset(newGrid);
        GRID.gridToSprite(newGrid, entity.actor);
        entity.actor.orientation = entity.actor.getOrientation(
            entity.moveState.dir
        );
        entity.actor.animateMove(entity.actor.orientation);

        if (changeView) {
            ENGINE.VIEWPORT.check(entity.actor);
        }
        ENGINE.VIEWPORT.alignTo(entity.actor);
        return;
    },
    teleportToGrid(entity, grid, changeView = false) {
        entity.moveState.reset(grid);
        GRID.gridToSprite(grid, entity.actor);
        if (changeView) {
            ENGINE.VIEWPORT.check(entity.actor);
        }
        ENGINE.VIEWPORT.alignTo(entity.actor);
    },
    gridToIndex(grid, map = MAP[GAME.level]) {
        return grid.x + grid.y * map.width;
    },
    indexToGrid(index, map = MAP[GAME.level]) {
        let x = index % map.width;
        let y = Math.floor(index / map.width);
        return new Grid(x, y);
    },
    vision(startGrid, endGrid, GA) {
        if (GRID.same(startGrid, endGrid)) return true;
        let path = GRID.raycasting(startGrid, endGrid);
        return GA.pathClear(path);
    },
    vision3D(startGrid, endGrid, GA) {
        if (GRID.same3D(startGrid, endGrid)) return true;
        let path = GRID.raycasting3D(startGrid, endGrid);
        return GA.pathClear(path);
    },
    freedom(startGrid, endGrid, IA) {
        if (GRID.same(startGrid, endGrid)) return true;
        let path = GRID.raycasting(startGrid, endGrid).slice(1);
        let candidates = IA.unrollArray(path);
        if (candidates.size > 0) {
            return false;
        } else return true;
    },
    freedom3D(startGrid, endGrid, IA) {
        if (GRID.same3D(startGrid, endGrid)) return true;
        let path = GRID.raycasting3D(startGrid, endGrid).slice(1);
        let candidates = IA.unrollArray(path);
        if (candidates.size > 0) {
            return false;
        } else return true;
    },
    sameFloor(startGrid, endGrid) {
        return startGrid.z === endGrid.z;
    },
    raycasting(startGrid, endGrid) {
        let normDir = startGrid.direction(endGrid);
        let path = [];
        path.push(Grid.toClass(startGrid));
        let x = startGrid.x;
        let y = startGrid.y;
        let dx = Math.abs(endGrid.x - x);
        let dy = -Math.abs(endGrid.y - y);
        let Err = dx + dy;
        let E2, node;
        do {
            E2 = Err * 2;
            if (E2 >= dy) {
                Err += dy;
                x += normDir.x;
            }
            if (E2 <= dx) {
                Err += dx;
                y += normDir.y;
            }
            node = new Grid(x, y);
            path.push(node);
        } while (!GRID.same(node, endGrid));
        return path;
    },
    raycasting3D(startGrid, endGrid) {
        let path = [];
        path.push(startGrid);

        let x = startGrid.x;
        let y = startGrid.y;
        let z = startGrid.z;

        let dx = endGrid.x - x;
        let dy = endGrid.y - y;
        let dz = endGrid.z - z;

        let stepX = Math.sign(dx);
        let stepY = Math.sign(dy);
        let stepZ = Math.sign(dz);

        dx = Math.abs(dx);
        dy = Math.abs(dy);
        dz = Math.abs(dz);

        let tMaxX = dx === 0 ? Infinity : 0.5 / dx;
        let tMaxY = dy === 0 ? Infinity : 0.5 / dy;
        let tMaxZ = dz === 0 ? Infinity : 0.5 / dz;

        let tDeltaX = dx === 0 ? Infinity : 1 / dx;
        let tDeltaY = dy === 0 ? Infinity : 1 / dy;
        let tDeltaZ = dz === 0 ? Infinity : 1 / dz;

        let node;

        while (x !== endGrid.x || y !== endGrid.y || z !== endGrid.z) {
            if (tMaxX < tMaxY) {
                if (tMaxX < tMaxZ) {
                    x += stepX;
                    tMaxX += tDeltaX;
                } else {
                    z += stepZ;
                    tMaxZ += tDeltaZ;
                }
            } else {
                if (tMaxY < tMaxZ) {
                    y += stepY;
                    tMaxY += tDeltaY;
                } else {
                    z += stepZ;
                    tMaxZ += tDeltaZ;
                }
            }

            node = new Grid3D(x, y, z);
            path.push(node);
        }

        return path;
    },
    calcDistancesBFS_BH(start, dungeon) {
        dungeon.setNodeMap();
        let BH = new BinHeap("distance");
        dungeon.GA.nodeMap[start.x][start.y].distance = 0;
        dungeon.GA.nodeMap[start.x][start.y].goto = new Vector(0, 0);
        BH.insert(dungeon.GA.nodeMap[start.x][start.y]);
        while (BH.size() > 0) {
            let node = BH.extractMax();
            for (let D = 0; D < ENGINE.directions.length; D++) {
                let nextNode = dungeon.GA.nodeMap[node.grid.x + ENGINE.directions[D].x][node.grid.y + ENGINE.directions[D].y];
                if (nextNode) {
                    if (nextNode.distance > node.distance + 1) {
                        nextNode.distance = node.distance + 1;
                        nextNode.prev = node.grid;
                        nextNode.goto = ENGINE.directions[D].mirror();
                        BH.insert(nextNode);
                    }
                }
            }
        }
    },
    calcDistancesBFS_A(start, dungeon, mode = GROUND_MOVE_GRID_EXCLUSION, nodeMap = "nodeMap") {
        dungeon.setNodeMap(mode, nodeMap);
        let Q = new NodeQ("distance");
        dungeon.GA[nodeMap][start.x][start.y].distance = 0;
        dungeon.GA[nodeMap][start.x][start.y].goto = new Vector(0, 0);
        Q.queueSimple(dungeon.GA[nodeMap][start.x][start.y]);
        while (Q.size() > 0) {
            let node = Q.dequeue();

            for (let D = 0; D < ENGINE.directions.length; D++) {
                let x = (node.grid.x + ENGINE.directions[D].x + dungeon.width) % dungeon.width;
                let y = (node.grid.y + ENGINE.directions[D].y + dungeon.height) % dungeon.height;
                let nextNode = dungeon.GA[nodeMap][x][y];

                if (nextNode) {
                    if (nextNode.distance > node.distance + 1) {
                        nextNode.distance = node.distance + 1;
                        nextNode.prev = node.grid;
                        nextNode.goto = ENGINE.directions[D].mirror();
                        Q.queueSimple(nextNode);
                    }
                }
            }
        }
    },
    calcDistancesBFS_A_3D(start, dungeon, _3D = false, mode = GROUND_MOVE_GRID_EXCLUSION, nodeMap = "nodeMap") {
        dungeon.GA.setNodeMap(nodeMap, mode, "exclude");

        let startNode = dungeon.GA[nodeMap][start.x][start.y][start.z];
        let exclusionNode = false;

        if (!startNode) {
            startNode = new PathNode3D(start.x, start.y, start.z, true);
            exclusionNode = true;
        }

        let Q = new NodeQ("distance");
        startNode.distance = 0;
        startNode.goto = NOWAY3;
        Q.queueSimple(startNode);

        const DIR = _3D ? [...ENGINE.directions3D] : [...ENGINE.directions3D_XY_plane];

        while (Q.size() > 0) {
            let node = Q.dequeue();

            for (let D = 0; D < DIR.length; D++) {

                let x = (node.grid.x + DIR[D].x + dungeon.width) % dungeon.width;
                let y = (node.grid.y + DIR[D].y + dungeon.height) % dungeon.height;
                let z = (node.grid.z + DIR[D].z + dungeon.depth) % dungeon.depth;
                let nextNode = dungeon.GA[nodeMap][x][y][z];

                if (nextNode) {
                    if (nextNode.distance > node.distance + 1) {
                        nextNode.distance = node.distance + 1;
                        nextNode.prev = node.grid;
                        node.exclusion ? nextNode.goto = NOWAY3 : nextNode.goto = DIR[D].mirror(); //this prevens moving in exlusion zone using goto, but still provides  distance;

                        Q.queueSimple(nextNode);
                    }
                }
            }
        }

        if (exclusionNode) startNode = null;
    },
    pathFromNodeMap(origin, nodeMap) {
        let path = [origin];
        let prev = nodeMap[origin.x][origin.y].prev;
        while (prev) {
            path.push(prev);
            prev = nodeMap[prev.x][prev.y].prev;
        }
        return path;
    },
    pathFromNodeMap3D(origin, nodeMap) {
        let path = [origin];
        let prev = nodeMap[origin.x][origin.y][origin.z].prev;
        while (prev) {
            path.push(prev);
            prev = nodeMap[prev.x][prev.y][prev.z].prev;
        }
        return path;
    },
    directionsFromPath(path, cut = false) {
        let directions = [];
        let from = path.pop();
        while (path.length > 0) {
            directions.push(from.direction(path.last()));
            from = path.pop();
            if (cut && directions.length === cut) {
                return directions;
            }
        }
        return directions;
    },
    getReboundDir(innerPoint, outerPoint, dir, GA, depth) {
        const inner = Grid3D.addDepth(innerPoint, depth);
        const outer = Grid3D.addDepth(outerPoint, depth);

        if (GA.isWall(outer)) {
            console.error("Missile position in wall. This should never have happened! But it is handled.", outer, GA.isWall(outer));
            console.info("innerPoint", innerPoint, "outerPoint", outerPoint);
            return null;
        }
        let faceNormal = outer.sub(inner);
        let newDir;
        let reverseDir = dir.mirror();
        faceNormal = Vector.toClass(faceNormal);

        if (!faceNormal.isOrto()) {
            faceNormal = GRID.resolveCornerBlock(faceNormal, inner, GA);
        }

        if (!faceNormal.isOrto()) {
            newDir = FP_Vector.toClass(faceNormal).normalize();
        } else if (GRID.same(faceNormal, NOWAY)) {
            newDir = outerPoint.direction(innerPoint);
        } else {
            let angle = FP_Vector.toClass(faceNormal).radAngleBetweenVectorsSharp(reverseDir);
            newDir = faceNormal.rotate(-angle);
        }

        return newDir;
    },
    resolveCornerBlock(faceNormal, innerGrid, GA) {
        const clone = faceNormal.clone();
        if (GA.isWall(innerGrid.add(new Vector(faceNormal.x, 0)))) faceNormal.x = 0;
        if (GA.isWall(innerGrid.add(new Vector(0, faceNormal.y)))) faceNormal.y = 0;
        if (faceNormal.isNull()) return clone;
        return faceNormal;
    },

    /**
    * Processes an input map object by verifying its dimensions and converting each row of the map's grid 
    * into a binary string representation. The resulting binary strings are collected and returned as an array.
    * 
    * @param {Object} map - The map object to be processed.
    * @param {number} map.height - The height of the map.
    * @param {number} map.width - The width of the map.
    * @param {number[]} map.grid - An array of floating-point numbers representing each row in the map.
    * @returns {string[]} An array of binary string representations of each row in the map's grid. 
    *                     Each string is padded to match the map's width.
    * @throws Will throw an error if the height of the map does not match the length of the grid, 
    *         indicating that the map is corrupted.
    * 
    */
    unpackIntegerMap(map, trim = true) {
        const h = map.height;
        const w = map.width;
        if (h != map.grid.length) throw `Map corrupted: height: ${h} grid.length: ${map.grid.length}`;
        var binary = [];
        for (let i = 0; i < h; i++) {
            let bin = float64ToInt64Binary(map.grid[i]).padStart(w, "0");
            if (trim) bin = bin.slice(-w);
            binary.push(bin);
        }
        return binary;
    }
};

class PathNode {
    constructor(x, y) {
        this.distance = Infinity;
        this.priority = Infinity;
        this.path = Infinity;
        this.prev = null;
        this.goto = null;
        this.grid = new Grid(x, y);
        this.visited = false;
    }
    setPriority() {
        this.priority = this.path + this.distance;
    }
}

class PathNode3D {
    constructor(x, y, z, exclusion = false) {
        this.distance = Infinity;
        this.priority = Infinity;
        this.path = Infinity;
        this.prev = null;
        this.goto = null;
        this.grid = new Grid3D(x, y, z);
        this.visited = false;
        this.exclusion = exclusion;
    }
    setPriority() {
        this.priority = this.path + this.distance;
    }
}

class BinHeap {
    constructor(prop) {
        this.HEAP = [];
        this.sort = prop;
    }
    size() {
        return this.HEAP.length;
    }
    parent(i) {
        return Math.floor((i - 1) / 2);
    }
    leftChild(i) {
        return 2 * i + 1;
    }
    rightChild(i) {
        return 2 * i + 2;
    }
    siftUp(i) {
        while (
            i > 0 &&
            this.HEAP[this.parent(i)][this.sort] > this.HEAP[i][this.sort]
        ) {
            this.HEAP.swap(this.parent(i), i);
            i = this.parent(i);
        }
    }
    siftDown(i) {
        let maxIndex = i;
        let L = this.leftChild(i);
        if (
            L < this.size() &&
            this.HEAP[L][this.sort] < this.HEAP[maxIndex][this.sort]
        ) {
            maxIndex = L;
        }
        let R = this.rightChild(i);
        if (
            R < this.size() &&
            this.HEAP[R][this.sort] < this.HEAP[maxIndex][this.sort]
        ) {
            maxIndex = R;
        }
        if (i !== maxIndex) {
            this.HEAP.swap(i, maxIndex);
            this.siftDown(maxIndex);
        }
    }
    insert(node) {
        this.HEAP.push(node);
        this.siftUp(this.size() - 1);
    }
    extractMax() {
        let result = this.HEAP[0];
        this.HEAP[0] = this.HEAP[this.size() - 1];
        this.HEAP.pop();
        this.siftDown(0);
        return result;
    }
    display() {
        while (this.size() > 0) {
            console.log(this.extractMax());
        }
    }
}

class SearchNode {
    constructor(HG, goal, stack, path, history, iterations) {
        this.grid = HG;
        this.stack = stack || [];
        this.history = history || [HG];
        this.path = path || 0;
        this.dist = this.grid.distance(goal);
        this.priority = this.path + this.dist;
        this.status = "Progress";
        this.iterations = iterations || 0;
    }
    append(node, goal) {
        let stack = this.stack.concat(node.stack);
        let history = this.history.concat(node.history.slice(1));
        let path = this.path + node.path;
        return new SearchNode(node.grid, goal, stack, path, history);
    }
}

class BlindNode {
    constructor(HG, stack, path, history, iterations) {
        this.grid = HG;
        this.stack = stack || [];
        this.history = history || [HG];
        this.path = path || 0;
        this.status = "Progress";
        this.iterations = iterations || 0;
    }
}

class NodeQ {
    constructor(prop) {
        this.list = [];
        this.sort = prop;
    }
    dequeue() {
        return this.list.shift();
    }
    destack() {
        return this.list.pop();
    }
    size() {
        return this.list.length;
    }
    queueSimple(node) {
        var included = false;
        for (let q = 0; q < this.list.length; q++) {
            if (node[this.sort] < this.list[q][this.sort]) {
                this.list.splice(q, 0, node);
                included = true;
                break;
            }
        }
        if (!included) this.list.push(node);
    }
    queue(node) {
        var included = false;
        for (let q = 0; q < this.list.length; q++) {
            if (node.priority < this.list[q].priority) {
                this.list.splice(q, 0, node);
                included = true;
                break;
            } else if (
                node.priority === this.list[q].priority &&
                node.dist < this.list[q].dist
            ) {
                this.list.splice(q, 0, node);
                included = true;
                break;
            }
        }
        if (!included) this.list.push(node);
    }
}

const MAPDICT = {
    EMPTY: 0,                               //0
    WALL: 2 ** 0,                           //1
    ROOM: 2 ** 1,                           //2
    DOOR: 2 ** 2,                           //4 - inner door!

    //original - for random maps
    VACANT_PLACEHOLDER1: 2 ** 3,            //8
    BLOCKWALL: 2 ** 4,                      //16 - wall that is removable
    STAIR: 2 ** 5,                          //32
    SHRINE: 2 ** 6,                         //64

    //alternative1 - RUN scpecific
    TRAP_DOOR: 2 ** 3,                      //8
    VACANT_PLACEHOLDER3: 2 ** 5,            //32
    DEAD_END: 2 ** 6,                       //64

    //alternative2 - CCC generation
    GATE: 2 ** 5,                           //32 - STAIR alias -> route to another dungeon
    HOLE: 2 ** 7,                           //128

    //aliases
    WARP: 2 ** 5,                           //32 - STAIR alias -> route to another part of the dungeon

    //16 bit extension
    WALL2: 2 ** 8,
    WALL4: 2 ** 9,
    WALL6: 2 ** 10,
    WALL8: 2 ** 11,

    //unused
    UNUSED: 2 ** 12,

    //special
    FOG: 2 ** 15,                            //32768 - fog should remain largest
    WATER: 2 ** 15,                          //32768 - fog,water should remain largest!
    RESERVED: 2 ** 14,                       //16384
    START_POSITION: 2 ** 13,                 //8192
};

const WallSizeToHeight = (value) => {
    return (Math.log2(value) - 7) * 2;
};

const reverseDictionary = (dict) => {
    const reversed = {};
    for (const [key, value] of Object.entries(dict)) {
        reversed[value] = key; // Last occurrence wins since object keys are overwritten
    }
    return reversed;
}

const REVERSED_MAPDICT = reverseDictionary(MAPDICT);
const STAIRCASE_GRIDS = [MAPDICT.WALL2, MAPDICT.WALL4, MAPDICT.WALL6, MAPDICT.WALL8];
const GROUND_MOVE_GRID_EXCLUSION = [MAPDICT.WALL, MAPDICT.HOLE, MAPDICT.BLOCKWALL, ...STAIRCASE_GRIDS];
const HERO_GROUND_MOVE_GRID_EXCLUSION = [MAPDICT.WALL, MAPDICT.HOLE, MAPDICT.BLOCKWALL];
const NO_FLY = [MAPDICT.WALL8, MAPDICT.WALL6];
const AIR_MOVE_GRID_EXCLUSION = [MAPDICT.WALL, MAPDICT.BLOCKWALL, ...NO_FLY];
const EXPLOADABLES = [MAPDICT.BLOCKWALL, MAPDICT.DOOR];
const ITEM_DROP_EXCLUSION = [MAPDICT.WALL, MAPDICT.BLOCKWALL];
const JUMP_MOVE = [MAPDICT.EMPTY, MAPDICT.HOLE, ...STAIRCASE_GRIDS];
const PATH_GRIDS = [MAPDICT.EMPTY, ...STAIRCASE_GRIDS];

class ArrayBasedDataStructure {
    constructor() { }
    indexToGrid(index) {
        return new Grid(index % this.width, index / this.width | 0);
    }
    indexTo2DGridSlice(index, z) {
        /** compatibility with 3d */
        return this.indexToGrid(index);
    }
    assertBounds(grid) {
        if (this.isOutOfBounds(grid)) throw new Error(`Grid is out of bounds: ${JSON.stringify(grid)}`);
    }
    gridToIndex(grid) {
        this.assertBounds(grid);
        return grid.x + grid.y * this.width;
    }
    isOut(grid) {
        return grid.x > this.maxX || grid.x < this.minX || grid.y > this.maxY || grid.y < this.minY;
    }
    isOutOfBounds(grid) {
        return grid.x < 0 || grid.x >= this.width || grid.y < 0 || grid.y >= this.height;
    }
    outside(grid) {
        return this.isOutOfBounds(grid);
    }
}

class GA_Dimension_Agnostic_Methods {
    constructor() { }
    massSet(bin) {
        for (let i = 0; i < this.map.length; i++) {
            this.map[i] |= bin;
        }
    }
    massReset(bin) {
        for (let i = 0; i < this.map.length; i++) {
            this.map[i] &= (2 ** this.gridSizeBit - 1 - bin);
        }
    }
    massClear() {
        this.map.fill(0);
    }
    sliceFill(start, len, fill) {
        for (let i = 0; i < len; i++) {
            this.map[start + i] = fill;
        }
    }
    linkToEntity(entities) {
        for (const entity of entities) {
            entity.moveState.gridArray = this;
        }
    }
    iset(index, bin) {
        this.map[index] |= bin;
    }
    set(grid, bin) {
        this.assertBounds(grid);
        this.map[this.gridToIndex(grid)] |= bin;
    }
    setValue(grid, value) {
        this.assertBounds(grid);
        this.map[this.gridToIndex(grid)] = value;
    }
    iclear(index, bin) {
        this.map[index] &= (2 ** this.gridSizeBit - 1 - bin);
    }
    clear(grid, bin) {
        this.assertBounds(grid);
        this.map[this.gridToIndex(grid)] &= (2 ** this.gridSizeBit - 1 - bin);
    }
    icheck(index, bin) {
        return this.map[index] & bin;
    }
    check(grid, bin) {
        if (this.isOutOfBounds(grid)) return false;
        return this.map[this.gridToIndex(grid)] & bin;
    }
    just_check(grid, bin) {
        return this.map[this.gridToIndex(grid)] & bin;
    }
    value(grid, value) {
        if (this.isOutOfBounds(grid)) return false;
        return this.map[this.gridToIndex(grid)] === value;
    }
    getValue(grid) {
        if (this.isOutOfBounds(grid)) return false;
        return this.map[this.gridToIndex(grid)];
    }
    iget_and_mask(index, not = 0) {
        return this.map[index] & (2 ** this.gridSizeBit - 1 - not);
    }
    toWall(grid) {
        this.setValue(grid, MAPDICT.WALL);
    }
    toHole(grid) {
        this.setValue(grid, MAPDICT.HOLE);
    }
    carveDot(grid) {
        this.setValue(grid, MAPDICT.EMPTY);
    }
    isWall(grid) {
        return this.check(grid, MAPDICT.WALL) === MAPDICT.WALL;
    }
    isHole(grid) {
        return this.check(grid, MAPDICT.HOLE) === MAPDICT.HOLE;
    }
    notWall(grid) {
        return !this.isWall(grid);
    }
    notHole(grid) {
        return !this.isHole(grid);
    }
    isMazeWall(grid) {
        if (this.isOutOfBounds(grid)) return false;
        return this.map[this.gridToIndex(grid)] & (MAPDICT.WALL === MAPDICT.WALL);
    }
    addDoor(grid) {
        this.set(grid, MAPDICT.DOOR);
    }
    toDoor(grid) {
        this.setValue(grid, MAPDICT.DOOR);
    }
    isDoor(grid) {
        return this.check(grid, MAPDICT.DOOR) === MAPDICT.DOOR;
    }
    notDoor(grid) {
        return !this.isDoor(grid);
    }
    addStair(grid) {
        this.set(grid, MAPDICT.STAIR);
        this.reserve(grid);
    }
    toStair(grid) {
        this.setValue(grid, MAPDICT.STAIR);
        this.reserve(grid);
    }
    isStair(grid) {
        return this.check(grid, MAPDICT.STAIR) === MAPDICT.STAIR;
    }
    notStair(grid) {
        return !this.isStair(grid);
    }
    closeDoor(grid) {
        if (this.isDoor(grid)) {
            this.set(grid, MAPDICT.WALL);
            this.reserve(grid);
        }
    }
    openDoor(grid) {
        if (this.isDoor(grid)) {
            this.clear(grid, MAPDICT.WALL + MAPDICT.DOOR + MAPDICT.RESERVED);
        }
    }
    isDoorClosed(grid) { }
    isDoorOpen(grid) { }
    toRoom(grid) {
        this.setValue(grid, MAPDICT.ROOM);
    }
    addRoom(grid) {
        this.set(grid, MAPDICT.ROOM);
    }
    isRoom(grid) {
        return this.check(grid, MAPDICT.ROOM) === MAPDICT.ROOM;
    }
    notRoom(grid) {
        return !this.isRoom(grid);
    }
    toShrine(grid) {
        this.setValue(grid, MAPDICT.SHRINE);
        this.reserve(grid);
    }
    addShrine(grid) {
        this.set(grid, MAPDICT.SHRINE);
        this.reserve(grid);
    }
    reserve(grid) {
        this.set(grid, MAPDICT.RESERVED);
    }
    isReserved(grid) {
        return this.check(grid, MAPDICT.RESERVED) === MAPDICT.RESERVED;
    }
    notReserved(grid) {
        return !this.isReserved(grid);
    }
    addTrapDoor(grid) {
        this.set(grid, MAPDICT.TRAP_DOOR);
    }
    isTrapDoor(grid) {
        return this.check(grid, MAPDICT.TRAP_DOOR) === MAPDICT.TRAP_DOOR;
    }
    notTrapDoor(grid) {
        return !this.isTrapDoor(grid);
    }
    toBlockWall(grid) {
        this.setValue(grid, MAPDICT.BLOCKWALL);
    }
    addBlockWall(grid) {
        this.set(grid, MAPDICT.BLOCKWALL);
    }
    isBlockWall(grid) {
        return this.check(grid, MAPDICT.BLOCKWALL) === MAPDICT.BLOCKWALL;
    }
    notBlockWall(grid) {
        return !this.isBlockWall(grid);
    }
    toEmpty(grid) {
        this.setValue(grid, MAPDICT.EMPTY);
    }
    isEmpty(grid) {
        return this.check(grid, MAPDICT.WALL) === MAPDICT.EMPTY;
    }
    isZero(grid) {
        return this.getValue(grid) === 0;
    }
    isValue(grid, value) {
        return this.getValue(grid) === value;
    }
    notEmpty(grid) {
        return !this.isEmpty(grid);
    }
    setStartPosition(grid) {
        this.set(grid, MAPDICT.START_POSITION);
    }
    isStartPosition(grid) {
        return this.check(grid, MAPDICT.START_POSITION) === MAPDICT.START_POSITION;
    }
    isFog(grid) {
        return this.check(grid, MAPDICT.FOG) === MAPDICT.FOG;
    }
    clearFog(grid) {
        this.clear(grid, MAPDICT.FOG);
    }
    setStackValue(stack, value) {
        for (const grid of stack) {
            this.setValue(grid, value);
        }
    }
    toString(clear = null) {
        const offset = 65;
        let str = "";
        for (let byte of this.map) {
            if (clear) {
                byte &= (2 ** this.gridSizeBit - 1 - clear);
            }
            str += String.fromCharCode(byte + offset);
        }
        return str;
    }
    exportMap() {
        return BWT.rle_encode(BWT.bwt(this.toString()));
    }

    /**
       * @param {FP_Grid} pos - position of entity
       * @param {number} exlusion - binary constant array of MAP_DICT values, @default GROUND_MOVE_GRID_EXCLUSION
       * @returns {boolean} - true if position is not in wall or other exluded type
       */
    entityNotInExcusion(pos, dir, r, depth = 0, exclusion = GROUND_MOVE_GRID_EXCLUSION, resolution = 8) {
        let checks = this.pointsAroundEntity(pos, dir, r, resolution);
        for (const point of checks) {
            let notExcluded = this.positionIsNotExcluded(point, exclusion, depth);
            if (!notExcluded) return false;
        }
        return true;
    }
    pointsAroundEntity(pos, dir, r, resolution = 4) {
        let checks = [];
        const increment = (2 * Math.PI) / resolution;
        for (let theta = 0; theta < 2 * Math.PI; theta += increment) {
            checks.push(pos.translate(dir.rotate(theta), r));
        }
        return checks;
    }
    forwardPointsFrontEntity(pos, dir, r, resolution = GRID.SETTING.FORWARD_CIRCLE_RESOLUTION) {
        let checks = [pos];
        const increment = (GRID.SETTING.FORWARD_CIRCLE_CHECK_ANGLE) / resolution;
        for (let i = 1; i <= resolution; i++) {
            let theta = increment * i;
            checks.push(pos.translate(dir.rotate(theta), r));
            checks.push(pos.translate(dir.rotate(-theta), r));
        }
        return checks;
    }
    pathClear(path) {
        for (const grid of path) {
            if (this.check(grid, AIR_MOVE_GRID_EXCLUSION.sum())) return false;
        }
        return true;
    }
    entityNotInWall(pos, dir, r, depth = 0, resolution = 8) {
        let checks = this.pointsAroundEntity(pos, dir, r, resolution);
        for (const point of checks) {
            let notWall = this.positionIsNotWall(point, depth);
            if (!notWall) return false;
        }
        return true;
    }
    getDirections(grid, value, leaveOut = null) {
        var directions = [];
        for (let D = 0; D < ENGINE.directions.length; D++) {
            if (leaveOut === null || !leaveOut.same(ENGINE.directions[D])) {
                let newGrid = grid.add(ENGINE.directions[D]);
                if (this.outside(newGrid)) {
                    if (GRID.SETTING.ALLOW_CROSS) {
                        newGrid = this.toOtherSide(newGrid);
                    } else continue;
                }
                if (this.value(newGrid, value)) {
                    directions.push(ENGINE.directions[D]);
                }
            }
        }
        return directions;
    }
}

class GridArray extends Classes([ArrayBasedDataStructure, GA_Dimension_Agnostic_Methods]) {
    constructor(sizeX, sizeY, byte = 1, fill = 0) {
        super();
        if (![1, 2, 4].includes(byte)) {
            console.error("GridArray set up with wrong size. Reset to default 8 bit!");
            byte = 1;
        }
        const byteToType = {
            1: Uint8Array,
            2: Uint16Array,
            4: Uint32Array
        };
        const GM = new (byteToType[byte])(sizeX * sizeY);
        this.width = parseInt(sizeX, 10);
        this.height = parseInt(sizeY, 10);
        this.maxX = this.width - 2;
        this.maxY = this.height - 2;
        this.minX = 1;
        this.minY = 1;
        this.map = GM;
        this.nodeMap = null;
        this.byte = byte;
        this.gridSizeBit = byte * 8;
        if (fill !== 0) this.map.fill(fill);
    }
    border(width = 1, set = MAPDICT.WALL) {
        this.rect(0, 0, this.width, this.height, width, set);
    }
    rect(X, Y, W, H, width = 1, set = MAPDICT.WALL) {
        for (let x = X; x < X + W; x++) {
            for (let w = 0; w < width; w++) {
                let grid1 = new Grid(x, Y + w);
                let grid2 = new Grid(x, Y + H - 1 - w);
                this.set(grid1, set);
                this.set(grid2, set);
            }
        }
        for (let y = Y; y < Y + H; y++) {
            for (let w = 0; w < width; w++) {
                let grid1 = new Grid(X + w, y);
                let grid2 = new Grid(X + W - 1 - w, y);
                this.set(grid1, set);
                this.set(grid2, set);
            }
        }
    }
    importMap(map) {
        this.map = map;
    }
    importGridMap(map) {
        /** map is maze or dungeon object */
        for (let y = 0; y < map.height; y++) {
            for (let x = 0; x < map.width; x++) {
                let grid = new Grid(x, y);
                if (map.isMazeWall(grid)) {
                    this.toWall(grid);
                }
            }
        }
    }
    importBinaryString(bin) {
        /** array of binary string */
        console.assert(bin.length === this.height, "Binary string not matching GA height");
        console.assert(bin[0].length === this.width, "Binary string not matching GA width");
        for (let [rowIndex, rowString] of bin.entries()) {
            for (let [columnIndex, char] of Array.from(rowString).entries()) {
                if (char === "0") {
                    let grid = new Grid(columnIndex, rowIndex);
                    this.carveDot(grid);
                }
            }
        }
    }
    toOtherSide(grid) {
        grid.x = (grid.x + this.width) % this.width;
        grid.y = (grid.y + this.height) % this.height;
        return grid;
    }
    checkLine(start, dir, length, value) {
        while (length > 0) {
            start = start.add(dir);
            if (this.isOutOfBounds(start)) return false;
            if (!this.value(start, value)) return false;
            length--;
        }
        return true;
    }
    checkNext(start, dir, value) {
        let next = start.add(dir);
        if (this.isOutOfBounds(next)) return false;
        return this.value(next, value);
    }
    setNodeMap(where = "nodeMap", path = [0], type = "value", block = [], cls = PathNode) {
        const map = new Array(this.width);
        const pathSum = path.sum();

        for (let x = 0; x < this.width; x++) {
            const mapX = map[x] = new Array(this.height);

            for (let y = 0; y < this.height; y++) {
                const grid = new Grid(x, y);

                let carve;
                switch (type) {
                    case "value":
                        carve = path.includes(this.map[this.gridToIndex(grid)]);
                        break;
                    case "exclude":
                        carve = !this.check(grid, pathSum);
                        break;
                    case "include":
                        carve = this.check(grid, pathSum);
                        break;
                    default:
                        console.error("nodeMap type error!");
                        return;
                }

                if (carve) {
                    mapX[y] = new cls(x, y);
                } else {
                    mapX[y] = null;
                }
            }
        }

        for (const obj of block) {
            map[obj.x][obj.y] = null;
        }

        this[where] = map;
        return map;
    }
    getDirectionsFromNodeMap(grid, nodeMap, leaveOut = null, allowCross = false) {
        var directions = [];
        for (let D = 0; D < ENGINE.directions.length; D++) {
            if (leaveOut === null || !leaveOut.same(ENGINE.directions[D])) {
                let newGrid = grid.add(ENGINE.directions[D]);

                if (this.outside(newGrid)) {
                    if (allowCross) {
                        newGrid = this.toOtherSide(newGrid);
                    } else continue;
                }

                if (nodeMap[newGrid.x][newGrid.y]) {
                    directions.push(ENGINE.directions[D]);
                }
            }
        }
        return directions;
    }
    directionsFromVisitedNodeMap(grid, nodeMap, allowCross = false) {
        var directions = [];
        for (let D = 0; D < ENGINE.directions.length; D++) {
            let newGrid = grid.add(ENGINE.directions[D]);

            if (this.outside(newGrid)) {
                if (allowCross) {
                    newGrid = this.toOtherSide(newGrid);
                } else continue;
            }

            if (nodeMap[newGrid.x][newGrid.y]) {
                if (!nodeMap[newGrid.x][newGrid.y].visited) {
                    directions.push(ENGINE.directions[D]);
                }
            }
        }
        return directions;
    }
    findAllNodesOnPath(start, finish, path = [0]) {
        let STACK = [];
        let NODES = [];
        let NodeMap = this.setNodeMap("AllNodes", path);
        STACK.push(start);
        while (STACK.length > 0) {
            let node = STACK.pop();
            NODES.push(node);
            NodeMap[node.x][node.y].visited = true;
            if (GRID.same(node, finish)) continue;
            let dirs = this.directionsFromVisitedNodeMap(node, NodeMap);
            for (const d of dirs) {
                STACK.push(node.add(d));
            }
        }
        return NODES;
    }
    gravity_AStar(start, finish, path = [0], type = "value", block = [], DIR = [LEFT, RIGHT, DOWN]) {
        return this.findPath_AStar_fast(start, finish, path, type, block, DIR);
    }
    findPath_AStar_fast(start, finish, path = [0], type = "value", block = [], DIR = ENGINE.directions) {
        /** 
        DIR: applicable directions
        return
        null: no path exist
        0: start is the same as finish
        nodeMap: path found, extract from nodeMap
        */

        if (GRID.same(start, finish)) {
            return 0;
        }

        var Q = new NodeQ("priority");
        let NodeMap = this.setNodeMap("AStar", path, type, block);

        NodeMap[start.x][start.y].distance = start.distance(finish);
        NodeMap[start.x][start.y].path = 0;
        NodeMap[start.x][start.y].setPriority();

        Q.queueSimple(NodeMap[start.x][start.y]);
        while (Q.size() > 0) {
            let node = Q.dequeue();
            for (let D = 0; D < DIR.length; D++) {
                let x = (node.grid.x + DIR[D].x + this.width) % this.width;
                let y = (node.grid.y + DIR[D].y + this.height) % this.height;

                let nextNode = NodeMap[x][y];
                if (nextNode) {
                    if (nextNode.path > node.path + 1) {
                        nextNode.path = node.path + 1;
                        nextNode.prev = node.grid;
                        nextNode.distance = nextNode.grid.distance(finish);
                        nextNode.setPriority();
                        Q.queueSimple(nextNode);
                        if (nextNode.distance === 0) {
                            return NodeMap;
                        }
                    }
                }
            }
        }
        return null;
    }
    floodFill(grid, value, condition = [0]) {
        var Q = [grid];
        let NodeMap = this.setNodeMap("floodFillNodeMap", condition);
        var selected;
        let iterations = 0;
        while (Q.length > 0) {
            selected = Q.shift();
            let dirs = this.getDirectionsFromNodeMap(selected, NodeMap);
            for (let q = 0; q < dirs.length; q++) {
                let next = selected.add(dirs[q]);
                NodeMap[next.x][next.y] = null;
                Q.push(next);
            }
            this.setValue(selected, value);
            iterations++;
        }
        return iterations;
    }
    floodFillSearch(grid, search, condition = [0]) {
        var Q = [grid];
        let NodeMap = this.setNodeMap("floodFillNodeMap", condition);
        var selected;
        while (Q.length > 0) {
            selected = Q.shift();
            let dirs = this.getDirectionsFromNodeMap(selected, NodeMap);
            for (let q = 0; q < dirs.length; q++) {
                let next = selected.add(dirs[q]);
                NodeMap[next.x][next.y] = null;
                Q.push(next);
            }
            if (selected.x === search.x && selected.y === search.y) {
                return true;
            }
        }
        return false;
    }
    getDirectionsIfNot(grid, value, leaveOut = null) {
        var directions = [];
        for (let D = 0; D < ENGINE.directions.length; D++) {
            if (leaveOut === null || !leaveOut.same(ENGINE.directions[D])) {
                let newGrid = grid.add(ENGINE.directions[D]);

                if (!this.check(newGrid, value)) {
                    directions.push(ENGINE.directions[D]);
                }
            }
        }
        return directions;
    }
    findPathByValue(grid, follow, find, firstDir) {
        let NodeMap = this.setNodeMap("searchValue", [follow, find]);
        let back = grid.add(firstDir.mirror());
        NodeMap[back.x][back.y].visited = true;
        let Q = [new BlindNode(grid)];
        let iteration = 0;
        while (true) {
            let T = [];
            for (const q of Q) {
                if (this.value(q.grid, find)) {
                    q.status = "Found";
                    return q;
                }
                NodeMap[q.grid.x][q.grid.y].visited = true;
                let dirs = this.directionsFromVisitedNodeMap(q.grid, NodeMap);
                for (const dir of dirs) {
                    let nextGrid = q.grid.add(dir);
                    let history = [].concat(q.history);
                    history.push(nextGrid);
                    let dirStack = [].concat(q.stack);
                    dirStack.push(dir);
                    let fork = new BlindNode(nextGrid, dirStack, q.path + 1, history, iteration);
                    T.push(fork);
                }
            }
            Q = T;
            iteration++;
        }
    }
    cutPath(path, goodValue) {
        let start = 0;
        let end = path.length - 1;
        let mid;
        while (true) {
            mid = Math.floor((end + start) / 2);
            if (mid === start) {
                return path[mid];
            }
            if (this.value(path[mid], goodValue)) {
                start = mid;
            } else {
                end = mid;
            }
        }
    }
    followPathUntil(path, goodValue) {
        for (let i = 1; i < path.length; i++) {
            if (!this.value(path[i], goodValue)) {
                return path[i - 1];
            }
        }
        return path.last();
    }
    xFilterNodes(nodes, badValue) {
        let goodNodes = [];
        for (const node of nodes) {
            let ok = true;
            for (const dir of ENGINE.corners) {
                let check = node.add(dir);
                if (this.value(check, badValue)) {
                    ok = false;
                    break;
                }
            }
            if (ok) {
                goodNodes.push(node);
            }
        }
        return goodNodes;
    }
    findNextCrossroad(start, dir, fly) {
        let exlusion = GROUND_MOVE_GRID_EXCLUSION.sum();
        if (fly) {
            exlusion = AIR_MOVE_GRID_EXCLUSION.sum();
        }
        let directions = this.getDirectionsIfNot(start, exlusion, dir.mirror());
        //console.log("....findNextCrossroad", start, dir, directions);
        let lastDir = dir;
        while (directions.length <= 1) {
            if (directions.length === 0) return [null, null]; //dead end!
            start = start.add(directions[0]);
            directions = this.getDirectionsIfNot(start, exlusion, directions[0].mirror());
        }
        return [start, lastDir];
    }
    entityInWallPoint(pos, dir, r, resolution = 8) {
        let checks = this.pointsAroundEntity(pos, dir, r, resolution);
        for (const point of checks) {
            let isWall = !this.positionIsNotWall(point);
            if (isWall) return [true, point];
        }
        return [false, null];
    }
    gridsAroundEntity(pos, dir, r, resolution = 4) {
        let checks = this.pointsAroundEntity(pos, dir, r, resolution);
        checks = checks.filter(this.positionIsNotWall, this);
        return checks.map(Grid.toClass);
    }

    /**
    * this is 2D Grid specific
    */
    positionIsNotWall(pos) {
        const grid = Grid.toClass(pos);
        const check = this.check(grid, AIR_MOVE_GRID_EXCLUSION.sum());
        return !check;
    }

    /**
     * this is 2D Grid specific
     */
    positionIsNotExcluded(pos, exclusion = GROUND_MOVE_GRID_EXCLUSION) {
        const grid = Grid.toClass(pos);
        const check = this.check(grid, exclusion.sum());
        return !check;
    }
    lookForGrid(startGrid, dir, lookGrid) {
        do {
            startGrid = startGrid.add(dir);
            if (this.isWall(startGrid)) return false;
        } while (!startGrid.same(lookGrid));
        return true;
    }
    static importMap(rle) {
        return BWT.inverseBwt(BWT.rle_decode(rle));
    }
    static fromString(sizeX, sizeY, string, byte = 1) {
        const offset = 65;
        let GA = new GridArray(sizeX, sizeY, byte);
        for (let i = 0; i < string.length; i++) {
            GA.map[i] = string[i].charCodeAt(0) - offset;
        }
        return GA;
    }
    toTextureMap() {
        /** 0 - light can pass through */
        const paddedWidth = POT(this.width);
        const paddedHeight = POT(this.height);
        const pixelData = new Uint8Array(paddedWidth * paddedHeight).fill(255);

        for (let y = 0; y < this.height; y++) {
            for (let x = 0; x < this.width; x++) {
                const grid = new Grid(x, y);
                const index = y * paddedWidth + x;
                pixelData[index] = this.notWall(grid) ? 0 : 255;
            }
        }
        return pixelData;
    }
    clone(){
        const clone = new GridArray(this.width, this.height, this.byte);
        clone.map.set(this.map);                                                            // deep copy of the typed array
        return clone;
    }
}

class NodeArray extends ArrayBasedDataStructure {
    constructor(GA, CLASS, path = [0], ignore = [], type = 'value') {
        super();
        /**
         * always constructed from GridArray
         */
        this.width = GA.width;
        this.height = GA.height;
        this.map = Array.from({ length: this.width * this.height }, () => null);
        const ignoreSum = 2 ** GA.gridSizeBit - 1 - ignore.sum();
        for (let [index, _] of this.map.entries()) {
            let check = GA.map[index] & ignoreSum;
            let carve;
            switch (type) {
                case 'value':
                    carve = path.includes(check);
                    break;
                default:
                    console.error("NodeArray type error!");
            }
            if (carve) {
                this.map[index] = new CLASS(index, this.indexToGrid(index));
            }
        }
    }
    G_set(grid, property, value) {
        this.I_set(this.gridToIndex(grid), property, value);
    }
    I_set(index, property, value) {
        this.map[index][property] = value;
    }
}

class IA_Dimension_Agnostic_Methods {
    constructor() { }
    validate(bank, indexValue) {
        if (bank >= this.banks || bank < 0) {
            throw new Error(`Illegal bank value. Expected value between 0 and ${this.banks - 1}, but got ${bank}.`);
        }
        if (indexValue > this.layerSize || indexValue <= 0) {
            throw new Error(`Illegal indexValue. Expected value between 1 and ${this.layerSize}, but got ${indexValue}.`);
        }
    }
    add(grid, indexValue, bank) {
        this.validate(bank, indexValue);
        let index = this.gridToIndex(grid);
        if (bank > 0) {
            indexValue = indexValue << (this.bankBitWidth * bank);
        }
        this.map[index] += indexValue;
    }
    next(grid, indexValue) {
        let bank = this.nextFreeBank(grid);
        if (bank !== -1) {
            this.add(grid, indexValue, bank);
        } else return -1;
    }
    nextFreeBank(grid) {
        const index = this.gridToIndex(grid);
        let layerValue = this.map[index];
        let bank = 0;
        const bankBitWidth = this.bankBitWidth;
        const banks = this.banks;
        while (layerValue > 0) {
            layerValue = layerValue >>> bankBitWidth;
            bank++;
            if (bank >= banks) {
                return -1;
            }
        }
        return bank;
    }
    unroll(grid) {
        const index = this.gridToIndex(grid);
        const items = [];
        let layerValue = this.map[index];
        for (let i = 0; i < this.banks; i++) {
            const current = layerValue & this.layerSize;
            if (current) items.push(current);
            layerValue >>>= this.bankBitWidth;
        }
        return items;
    }
    unrollArray(arr) {
        let items = [];
        for (let a of arr) {
            items.push(...this.unroll(a));
        }
        return new Set(items);
    }
    clear(grid) {
        this.map[this.gridToIndex(grid)] = 0;
    }
    empty(grid) {
        return this.map[this.gridToIndex(grid)] === 0;
    }
    set(grid, indexValue, bank) {
        if (bank === undefined) {
            this.map[this.gridToIndex(grid)] = indexValue;
        } else {
            this.clear(grid);
            this.validate(bank, indexValue);
            this.add(grid, indexValue, bank);
        }
    }
    get(grid) {
        return this.map[this.gridToIndex(grid)];
    }
    has(grid, indexValue) {
        let items = this.unroll(grid);
        return items.includes(indexValue);
    }
    removeIndex(grid, indexValue) {
        let before = this.unroll(grid);
        let after = before.filter((el) => el !== indexValue);
        this.clear(grid);
        for (const [i, a] of after.entries()) {
            this.add(grid, a, i);
        }
    }
    hasFreeBanks(grid) {
        if (this.empty(grid)) {
            return true;
        } else if (this.nextFreeBank(grid) !== -1) {
            return true;
        }
        return false;
    }
    setup(byte, banks, sizeX, sizeY, sizeZ = 1) {
        if (![1, 2, 4].includes(byte)) {
            console.error("IndexArray set up with wrong size. Reset to default 8 bit!");
            byte = 1;
        }
        banks = parseInt(banks, 10);
        if (banks <= 0 || banks > byte * 4) {
            console.error("Illegal value for banks. Set to default 4!");
            banks = byte * 4;
        }
        const byteToType = {
            1: Uint8Array,
            2: Uint16Array,
            4: Uint32Array
        };
        const GM = new byteToType[byte](sizeX * sizeY * sizeZ);

        this.gridSizeBit = byte * 8;
        this.map = GM;
        this.banks = banks;
        this.bankBitWidth = this.gridSizeBit / this.banks;
        this.layerSize = 2 ** this.bankBitWidth - 1;
    }
}

class IndexArray extends Classes([ArrayBasedDataStructure, IA_Dimension_Agnostic_Methods]) {
    constructor(sizeX = 1, sizeY = 1, byte = 1, banks = 1) {
        super();
        this.setup(byte, banks, sizeX, sizeY, 1);
        this.width = sizeX;
        this.height = sizeY;
    }
}

/**
 * 3D version of ArrayBasedDataStructures
 */
class ArrayBasedDataStructure3D {
    constructor() { }
    indexToGrid(index) {
        let z = Math.floor(index / (this.width * this.height));
        index = index % (this.width * this.height);
        let y = Math.floor(index / this.width);
        let x = index % this.width;
        return new Grid3D(x, y, z);
    }
    isOutOfBounds(grid) {
        return grid.x < 0 || grid.x >= this.width || grid.y < 0 || grid.y >= this.height || grid.z < 0 || grid.z >= this.depth;
    }
    outside(grid) {
        return this.isOutOfBounds(grid);
    }
    assertBounds(grid) {
        if (this.isOutOfBounds(grid)) throw new Error(`Grid is out of bounds: ${JSON.stringify(grid)}`);
    }
    isOut(grid) {
        /** is out of inner bounds */
        return grid.x > this.maxX || grid.x < this.minX || grid.y > this.maxY || grid.y < this.minY || grid.z > this.maxZ || grid.z < this.minZ;
    }
    gridToIndex(grid) {
        this.assertBounds(grid);
        return grid.x + grid.y * this.width + grid.z * this.width * this.height;
    }
    indexTo2DGridSlice(index, z) {
        let offset = this.width * this.height * z;
        index -= offset;
        if (index < 0) return null;
        let y1 = Math.floor(index / this.width);
        let y2 = y1 % this.height;
        if (y1 !== y2) return null;                             // it's not position on the slice
        const grid = new Grid(index % this.width, y2);
        return grid;
    }
}

class GridArray3D extends Classes([ArrayBasedDataStructure3D, GA_Dimension_Agnostic_Methods]) {
    constructor(sizeX, sizeY, sizeZ, byte = 2, fill = 0) {
        super();

        this.width = parseInt(sizeX, 10);
        this.height = parseInt(sizeY, 10);
        this.depth = parseInt(sizeZ, 10);
        this.maxX = this.width - 2;
        this.maxY = this.height - 2;
        this.maxZ = this.depth - 1;
        this.minX = 1;
        this.minY = 1;
        this.minZ = 0;

        if (![1, 2, 4].includes(byte)) {
            console.error("GridArray set up with wrong size. Reset to default 16 bit!");
            byte = 2;
        }
        const byteToType = {
            1: Uint8Array,
            2: Uint16Array,
            4: Uint32Array
        };
        const GridMapArray = new (byteToType[byte])(sizeX * sizeY * sizeZ);
        this.map = GridMapArray;
        this.nodeMap = null;
        this.gridSizeBit = byte * 8;
        if (fill !== 0) this.map.fill(fill);
    }
    static importMap(rle) {
        return BWT.inverseBwt(BWT.rle_decode(rle));
    }
    static fromString(sizeX, sizeY, sizeZ, string, byte = 2) {
        const offset = 65;
        let GA = new GridArray3D(sizeX, sizeY, sizeZ, byte);
        for (let i = 0; i < string.length; i++) {
            GA.map[i] = string[i].charCodeAt(0) - offset;
        }
        return GA;
    }
    toTextureMap() {
        /** 
         * 0 - light can pass through 
         * POT not required for WebGL2
         **/

        const W = this.width;
        const H = this.height;
        const D = this.depth;
        const pixelData = new Uint8Array(W * H * D).fill(255);

        for (let z = 0; z < this.depth; z++) {
            for (let y = 0; y < this.height; y++) {
                for (let x = 0; x < this.width; x++) {
                    const grid = new Grid3D(x, y, z);
                    const index = z * W * H + y * W + x;
                    pixelData[index] = this.notWall(grid) ? 0 : 255;
                }
            }
        }

        return pixelData;
    }
    border(width = 1, floor = 0, set = MAPDICT.WALL) {
        this.rect(0, 0, this.width, this.height, width, floor, set);
    }
    rect(X, Y, W, H, width = 1, floor = 0, set = MAPDICT.WALL) {
        for (let x = X; x < X + W; x++) {
            for (let w = 0; w < width; w++) {
                let grid1 = new Grid3D(x, Y + w, floor);
                let grid2 = new Grid3D(x, Y + H - 1 - w, floor);
                this.setValue(grid1, set);
                this.setValue(grid2, set);
            }
        }
        for (let y = Y; y < Y + H; y++) {
            for (let w = 0; w < width; w++) {
                let grid1 = new Grid3D(X + w, y, floor);
                let grid2 = new Grid3D(X + W - 1 - w, y, floor);
                this.setValue(grid1, set);
                this.setValue(grid2, set);
            }
        }
    }
    fillArea(X, Y, W, H, floor, set) {
        for (let x = X; x <= X + W; x++) {
            for (let y = Y; y <= Y + H; y++) {
                let grid = new Grid3D(x, y, floor);
                this.setValue(grid, set);
            }
        }
    }
    gridsAroundEntity(pos, dir, r, depth, resolution = 4) {
        let checks = this.pointsAroundEntity(pos, dir, r, resolution);
        checks = checks.filter(pos => this.positionIsNotWall(pos, depth));
        return checks.map(pos => new Grid3D(pos.x, pos.y, depth));
    }
    positionIsNotExcluded(pos, exclusion = GROUND_MOVE_GRID_EXCLUSION, depth) {
        const grid = new Grid3D(pos.x, pos.y, depth);
        const check = this.check(grid, exclusion.sum());
        return !check;
    }
    positionIsNotWall(pos, depth) {
        const grid = new Grid3D(pos.x, pos.y, depth);
        const check = this.check(grid, AIR_MOVE_GRID_EXCLUSION.sum());
        return !check;
    }
    positionIsInInclusion(pos, depth, inclusion) {
        const grid = new Grid3D(pos.x, pos.y, depth);
        return this.check(grid, inclusion.sum());
    }
    setNodeMap(where = "nodeMap", path = [0], type = "value", block = [], cls = PathNode3D) {
        const pathSum = path.sum();
        const map = Array.from({ length: this.width }, (_, x) =>
            Array.from({ length: this.height }, (_, y) =>
                Array.from({ length: this.depth }, (_, z) => {
                    const grid = new Grid3D(x, y, z);

                    const carveTypes = {
                        value: path.includes(this.map[this.gridToIndex(grid)]),
                        exclude: !this.check(grid, pathSum),
                        include: this.check(grid, pathSum)
                    };

                    return carveTypes[type] ? new cls(x, y, z) : null;
                })
            )
        );

        for (const obj of block) {
            map[obj.x][obj.y][obj.z] = null;
        }

        this[where] = map;
        return map;
    }
    findNextCrossroad(start, dir, fly) {
        let exlusion = GROUND_MOVE_GRID_EXCLUSION.sum();
        if (fly > 0) exlusion = AIR_MOVE_GRID_EXCLUSION.sum();

        let directions = this.getDirectionsIfNot(start, exlusion, fly, dir.mirror());
        let lastDir = dir;
        while (directions.length <= 1) {
            if (directions.length === 0) return [null, null]; //dead end!
            start = start.add(directions[0]);
            directions = this.getDirectionsIfNot(start, exlusion, fly, directions[0].mirror());
        }
        return [start, lastDir];
    }
    getDirectionsIfNot(grid, value, fly = false, leaveOut = null) {
        const directions = [];
        const DIR = fly > 0.0 ? [...ENGINE.directions3D] : [...ENGINE.directions3D_XY_plane];

        for (let D = 0; D < DIR.length; D++) {
            if (leaveOut === null || !leaveOut.same(DIR[D])) {
                let newGrid = grid.add(DIR[D]);

                if (this.isOutOfBounds(newGrid)) continue;
                if (!this.just_check(newGrid, value)) {

                    directions.push(DIR[D]);
                }
            }
        }
        return directions;
    }
    entityInWallPoint(pos, dir, r, depth, resolution = 8) {
        let checks = this.pointsAroundEntity(pos, dir, r, resolution);
        for (const point of checks) {
            let isWall = !this.positionIsNotWall(point, depth);
            if (isWall) return [true, point];
        }
        return [false, null];
    }
    spherePointsAroundCenter(pos, dir, r) {
        pos = pos.array;
        dir = dir.array;
        const points = [];

        const forwardPoint = glMatrix.vec3.create();
        glMatrix.vec3.scaleAndAdd(forwardPoint, pos, dir, r);                                       // First point directly ahead
        points.push(forwardPoint);

        let up = [0, 1, 0];                                                                         // Find orthogonal vectors to create a hemisphere around the direction vector
        let orthogonalVec1 = glMatrix.vec3.create();
        glMatrix.vec3.cross(orthogonalVec1, dir, up);

        if (glMatrix.vec3.length(orthogonalVec1) < 0.001) {
            up = [1, 0, 0];                                                                         // Handle the case when direction is (almost) parallel to 'up' vector
            glMatrix.vec3.cross(orthogonalVec1, dir, up);
        }

        glMatrix.vec3.normalize(orthogonalVec1, orthogonalVec1);

        const orthogonalVec2 = glMatrix.vec3.create();                                              // Second orthogonal vector
        glMatrix.vec3.cross(orthogonalVec2, dir, orthogonalVec1);
        glMatrix.vec3.normalize(orthogonalVec2, orthogonalVec2);

        const angles = [Math.PI / 2, Math.PI / 4];                                                  // Choose angles defining hemisphere coverage 

        for (let angle of angles) {                                                                 // Generate hemisphere points
            const cosAngle = Math.cos(angle);
            const sinAngle = Math.sin(angle);

            for (let rad of [0, Math.PI / 2, Math.PI, (3 * Math.PI) / 2]) {                         // Points around the circle at each angle

                const sideVec = glMatrix.vec3.create();                                             // Rotate around 'dir' vector
                glMatrix.vec3.scale(sideVec, orthogonalVec1, Math.cos(rad));                        // sideVec = orthogonalVec1 * cos(t) + orthogonalVec2 * sin(t)
                glMatrix.vec3.scaleAndAdd(sideVec, sideVec, orthogonalVec2, Math.sin(rad));

                const hemispherePoint = glMatrix.vec3.create();                                     // hemispherePoint = pos + dir * radius * cos(angle) + sideVec * radius * sin(angle)
                const forwardComponent = glMatrix.vec3.create();
                glMatrix.vec3.scale(forwardComponent, dir, r * cosAngle);
                glMatrix.vec3.scale(sideVec, sideVec, r * sinAngle);
                glMatrix.vec3.add(hemispherePoint, pos, forwardComponent);
                glMatrix.vec3.add(hemispherePoint, hemispherePoint, sideVec);
                points.push(hemispherePoint);
            }
        }

        return points;
    }
    sphereInWallPoint(pos, dir, r) {
        let checks = this.spherePointsAroundCenter(pos, dir, r);
        for (const point of checks) {
            const grid3d = new Grid3D(point[0], point[2], point[1]);
            const check = this.check(grid3d, AIR_MOVE_GRID_EXCLUSION.sum());                          //if >0  then hit, if false the in was OOB
            if (check === false || check > 0) return [true, Vector3.from_array(point)];
        }

        return [false, null];
    }
    getDirectionsFromNodeMap(grid, nodeMap, fly, leaveOut = null, allowCross = false) {
        const directions = [];
        const DIR = fly > 0.0 ? [...ENGINE.directions3D] : [...ENGINE.directions3D_XY_plane];

        for (let D = 0; D < DIR.length; D++) {
            if (leaveOut === null || !leaveOut.same(DIR[D])) {
                let newGrid = grid.add(DIR[D]);

                if (this.outside(newGrid)) {
                    if (allowCross) {
                        newGrid = this.toOtherSide(newGrid);
                    } else continue;
                }

                if (nodeMap[newGrid.x][newGrid.y][newGrid.z]) {
                    directions.push(DIR[D]);
                }
            }
        }
        return directions;
    }
    findPath_AStar_fast(start, finish, path = [0], type = "value", fly = 0, block = []) {
        /** 
        DIR: applicable directions
        return
        null: no path exist
        0: start is the same as finish
        nodeMap: path found, extract from nodeMap
        */

        if (GRID.same(start, finish)) {
            return 0;
        }

        var Q = new NodeQ("priority");
        let NodeMap = this.setNodeMap("AStar", path, type, block);

        const DIR = fly > 0.0 ? [...ENGINE.directions3D] : [...ENGINE.directions3D_XY_plane];

        NodeMap[start.x][start.y][start.z].distance = start.distance(finish);
        NodeMap[start.x][start.y][start.z].path = 0;
        NodeMap[start.x][start.y][start.z].setPriority();

        Q.queueSimple(NodeMap[start.x][start.y][start.z]);
        while (Q.size() > 0) {
            let node = Q.dequeue();
            for (let D = 0; D < DIR.length; D++) {
                let x = (node.grid.x + DIR[D].x + this.width) % this.width;
                let y = (node.grid.y + DIR[D].y + this.height) % this.height;
                let z = (node.grid.z + DIR[D].z + this.depth) % this.depth;

                let nextNode = NodeMap[x][y][z];
                if (nextNode) {
                    if (nextNode.path > node.path + 1) {
                        nextNode.path = node.path + 1;
                        nextNode.prev = node.grid;
                        nextNode.distance = nextNode.grid.distance(finish);
                        nextNode.setPriority();
                        //console.log("next node", nextNode);
                        Q.queueSimple(nextNode);
                        if (nextNode.distance === 0) {
                            return NodeMap;
                        }
                    }
                }
            }
        }
        return null;
    }
    singleForwardPositionIsIncluded(pos, dir, r, depth, include, resolution = GRID.SETTING.FORWARD_CIRCLE_RESOLUTION) {
        let checks = this.forwardPointsFrontEntity(pos, dir, r, resolution);
        checks = checks.map(pos => new Grid3D(pos.x, pos.y, depth));
        let unpassable = checks.filter(grid => this.check(grid, HERO_GROUND_MOVE_GRID_EXCLUSION.sum()));
        if (unpassable.length > 0) return null;

        let filtered = checks.filter(grid => this.check(grid, include.sum()));
        return filtered;
    }
    singleForwardPositionIsEmpty(pos, dir, r, depth, resolution = GRID.SETTING.FORWARD_CIRCLE_RESOLUTION) {
        return this.singleForwardPositionIsValue(pos, dir, r, depth, 0, resolution);
    }
    singleForwardPositionIsValue(pos, dir, r, depth, value, resolution = GRID.SETTING.FORWARD_CIRCLE_RESOLUTION) {
        let checks = this.forwardPointsFrontEntity(pos, dir, r, resolution);
        checks = checks.map(pos => new Grid3D(pos.x, pos.y, depth));
        let filtered = checks.filter(grid => this.isValue(grid, value));
        return filtered;
    }
    forwardPositionIsEmpty(pos, dir, r, depth, resolution = GRID.SETTING.FORWARD_CIRCLE_RESOLUTION) {
        return this.forwardPositionIsValue(pos, dir, r, depth, 0, resolution);
    }
    forwardPositionIsValue(pos, dir, r, depth, value, resolution = GRID.SETTING.FORWARD_CIRCLE_RESOLUTION) {
        let checks = this.forwardPointsFrontEntity(pos, dir, r, resolution);
        checks = checks.map(pos => new Grid3D(pos.x, pos.y, depth));
        let filtered = checks.filter(grid => this.isValue(grid, value));
        return checks.length === filtered.length;
    }
    forwardPositionAreIn(pos, dir, r, depth, values, resolution = GRID.SETTING.FORWARD_CIRCLE_RESOLUTION) {
        let checks = this.forwardPointsFrontEntity(pos, dir, r, resolution);
        checks = checks.map(pos => new Grid3D(pos.x, pos.y, depth));
        let filtered = checks.filter(grid => values.includes(this.getValue(grid)));
        return checks.length === filtered.length;
    }
    findSolidFloor(pos) {
        let posGrid = Vector3.to_FP_Grid3D(pos);
        let grid = Vector3.to_Grid3D(pos);
        let gridBelow = grid.add(BELOW3);

        //console.info("findSolidFloor", posGrid, grid, gridBelow);

        while (!this.check(gridBelow, ITEM_DROP_EXCLUSION.sum()) && gridBelow.z >= 0) {
            grid = gridBelow;
            gridBelow = grid.add(BELOW3);
            //console.log("...gridBelow", gridBelow);
        }

        const gridValue = REVERSED_MAPDICT[this.getValue(grid)];
        //console.warn("*** gridValue", gridValue, "grid", grid);

        switch (gridValue) {
            case "EMPTY":
                posGrid.z = grid.z;
                break;
            case "WALL2":
            case "WALL4":
            case "WALL6":
            case "WALL8":
                const heightOffset = parseInt(gridValue[4], 10) / 10;
                posGrid.z = grid.z + heightOffset;
                break;
            case "HOLE": return null;
            default:
                throw new Error(`grid type not supported ${gridValue}`);

        }
        return posGrid;
    }
    blockVisible(grid) {
        // A block is visible if at least one neighbor is NOT out of bounds or a wall (or a door).
        if (this.isDoor(grid)) return true;                                         //door has to be always visible
        return ENGINE.directions3D.some(dir => {
            const checkGrid = grid.add(dir);
            return !(this.isOutOfBounds(checkGrid) || this.isWall(checkGrid) || this.isDoor(checkGrid));
        });
    }
}

class IndexArray3D extends Classes([ArrayBasedDataStructure3D, IA_Dimension_Agnostic_Methods]) {
    constructor(sizeX = 1, sizeY = 1, sizeZ = 1, byte = 1, banks = 1) {
        super();
        this.setup(byte, banks, sizeX, sizeY, sizeZ);
        this.width = sizeX;
        this.height = sizeY;
        this.depth = sizeZ;
    }
}

//END
console.log(`%cGRID ${GRID.VERSION} loaded.`, GRID.CSS);