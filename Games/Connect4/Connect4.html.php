   <!-- copy start-->
   <div id="preload" class="hidden"></div>
    <div class="container my-5 p-5 cool_page">
        <div>
            <div id="load"></div>
            <div class="row win">
                <h1 id="title" class="C4"></h1>
                <p>Connect-4 is the intense battlefield where discs fall like rain and IQs scatter like
                    confetti - especially when Silly thinks diagonals are just tilted rows. Meanwhile, Smarty
                    plots six moves ahead, rolling its circuits in despair as Prophet refuses to play unless someone
                    compliments her pattern.
                </p>
                <p>Seriously? You don't know this game?</p>
                <hr class="crimson-hr">

                <p class="fw-bold"> Agents explained:</p>
                <p><span class="fw-bold">Human:</span> That's you! Or your mate. Takes input from the keyboard (1..7 for
                    column, ignores bad moves.).</p>
                <p><span class="fw-bold">Random:</span> Take completely random turn. Not much fun to play against this
                    one. But it is was very useful for testing.</p>
                <p><span class="fw-bold">Silly:</span> 3-step lookahead algorithm. No challenge to beat this
                    one. But it beats Random player all the time. What a surprise.</p>
                <p><span class="fw-bold">Smarty:</span> 5-step lookahead algorithm. This feels like a real human (if
                    slightly drunk)
                    opponent. It beats Silly in 82% of cases, the rest were ties.</p>
                <p><span class="fw-bold">Prophet:</span> 7-step lookahead algorithm. It's the maximum javascript can
                    take before blocking the thread. It takes 0.5-2 seconds to calculate the move. It beats Smarty in
                    74% of cases.</p>

                <hr class="crimson-hr">

                <div id="setup" class="row">
                    <div class="col-4">
                        <label for="select_player_start">
                            Which player starts first:
                        </label>
                        <select name="select_player_start" id="select_player_start">
                            <option value="random" selected="selected">Random</option>
                            <option value="red">Red</option>
                            <option value="blue">Blue</option>
                        </select>

                        <br>
                        <label for="animation_speed">Animation speed</label>
                        <input type="number" min="5" max="30" value="15" id="animation_speed">

                        <div>
                            <fieldset>
                                <legend>Mode:</legend>
                                <input type="radio" id="game_mode" value="game" name="mode" checked>
                                <label for="game_mode">Game</label>
                                <br>
                                <input type="radio" id="analyze_mode" value="analyze" name="mode">
                                <label for="analyze_mode">Analyze</label>
                                <br>
                                <label for="number_of_runs">Number of runs</label>
                                <input type="number" min="5" max="100" value="20" id="number_of_runs" disabled>
                                <p>Game analysis is found in the console.</p>
                                <p>Game will run as fast as possible, without any animation. Using analysis mode with
                                    human players is silly.</p>
                            </fieldset>
                        </div>
                    </div>
                    <div class="col-4">
                        <fieldset>
                            <legend>Red:</legend>
                            <label for="red_player_agents">Red player agent:</label>
                            <select name="red_player_agents" id="red_player_agents"></select>
                            <br>
                            <label for="red_player_name">Red player name:</label>
                            <input type="text" id="red_player_name" value="Red"></input>
                        </fieldset>
                    </div>
                    <div class="col-4">
                        <fieldset>
                            <legend>Blue:</legend>
                            <label for="blue_player_agents">Blue player agent:</label>
                            <select name="blue_player_agents" id="blue_player_agents"></select>
                            <br>
                            <label for="blue_player_name">Blue player name:</label>
                            <input type="text" id="blue_player_name" value="Blue"></input>
                        </fieldset>
                    </div>
                </div>
            </div>
        </div>


        <div class="row my-5">
            <div id="debug" class="section">
                <fieldset>
                    <legend>
                        Engine versions:
                    </legend>
                    Prototype LIB: <span id="lib_version"></span><br>
                    ENGINE: <span id="engine_version"></span><br>
                    GRID: <span id="grid_version"></span><br>
                </fieldset>
            </div>
        </div>

        <div>
            <p id="buttons">
            <div>
                <input type='button' id='toggleAbout' value='About'>
                <input type='button' id='toggleVersion' value='Version'>
                <input type='button' id='pause' value='Pause Game [F4]' disabled="disabled">
            </div>
            </p>
        </div>

        <div id="about" class="section">
            <fieldset>
                <legend>
                    About:
                </legend>
                <div class="row">
                    <div class="col-12 col-lg-3 my-2 d-flex align-items-center justify-content-center">
                        <image src="" alt="" class="img-fluid  border-dark rounded-2" title="">
                        </image>
                    </div>
                    <div class="col-12 col-lg-6 my-2">
                        <p>The game was written as a break from working on 'Haunting the Hauntessa', using my Python <a
                                href="https://www.kaggle.com/code/lovroselic/connectx-ls" target="_blank">submission to
                                Kaggle competition</a> as the source. One day, I will find the time to implement also
                            reinforced learning agent ... hopefully.</p>
                    </div>
                    <div class="col-12 col-lg-3 my-2 d-flex align-items-center justify-content-center">
                        <image src="" alt="" class="img-fluid  border-dark rounded-2" title="">
                        </image>
                    </div>
                </div>
            </fieldset>
        </div>

        <p class="version terminal" id="version"></p>
    </div>

    <div class="container">
        <div id="game" class="winTrans"></div>
        <div id="bottom" class="cb" style="margin-top: 1024px"></div>
        <div id="temp" class="hidden"></div>
        <div id="temp2" class="hidden"></div>
    </div>
    <!-- COPY END -->