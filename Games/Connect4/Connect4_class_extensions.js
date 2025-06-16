class C4Grid extends GridArray {
    //to simplify the syntax
    toRed(grid) {
        this.setValue(grid, MAPDICT.RED);
    }
    toBlue(grid) {
        this.setValue(grid, MAPDICT.BLUE);
    }
    setToken(grid, player) {
        switch (player) {
            case "red": return this.toRed(grid);
            case "blue": return this.toBlue(grid);
            default: throw `wrong player: ${player}`;
        }
    }
}