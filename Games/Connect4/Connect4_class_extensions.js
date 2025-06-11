class C4Grid extends GridArray {
    //to simplify the syntax
    toRed(grid) {
        this.setValue(grid, MAPDICT.RED);
    }
    toBlue(grid) {
        this.setValue(grid, MAPDICT.BLUE);
    }
}