"""Finish all TODO items in this file to complete the isolation project, then
test your agent's strength against a set of known agents using tournament.py
and include the results in your report.
"""
import random


class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass


def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    This should be the best heuristic function for your project submission.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """

    if game.is_winner(player):
        return float("inf")

    if game.is_loser(player):
        return float("-inf")


    p_moves = game.get_legal_moves(player)
    opp = game.get_opponent(player)
    opp_moves = game.get_legal_moves(opp)
    
    w, h = game.width / 2., game.height / 2.
    p_vals = 1
    for move in p_moves:
        y, x = move
        p_vals += float((h - y)**2 + (w - x)**2)

    o_vals = 1
    for move in opp_moves:
        y, x = move
        o_vals += float((h - y)**2 + (w - x)**2)

    return float(o_vals/p_vals)


def custom_score_2(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    if game.is_winner(player):
        return float("inf")

    if game.is_loser(player):
        return float("-inf")

    p_moves = game.get_legal_moves()
    opp = game.get_opponent(player)
    opp_moves = game.get_legal_moves(opp)

    return float((len(p_moves)+1)/(len(opp_moves)+1))


def custom_score_3(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    if game.is_winner(player):
        return float("inf")

    if game.is_loser(player):
        return float("-inf")

    opp = game.get_opponent(player)

    w, h = game.width / 2., game.height / 2.

    y, x = game.get_player_location(player)
    p_val = float((h - y)**2 + (w - x)**2)

    y, x = game.get_player_location(opp)
    o_val = float((h - y)**2 + (w - x)**2)

    return float(o_val+1 / p_val+1)



class IsolationPlayer:
    """Base class for minimax and alphabeta agents -- this class is never
    constructed or tested directly.

    ********************  DO NOT MODIFY THIS CLASS  ********************

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """
    def __init__(self, search_depth=3, score_fn=custom_score, timeout=10.):
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout


class MinimaxPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using depth-limited minimax
    search. You must finish and test this player to make sure it properly uses
    minimax to return a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        **************  YOU DO NOT NEED TO MODIFY THIS FUNCTION  *************

        For fixed-depth search, this function simply wraps the call to the
        minimax method, but this method provides a common interface for all
        Isolation agents, and you will replace it in the AlphaBetaPlayer with
        iterative deepening search.

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # initialize `best_move` as random available move
                # initialize `best_move` as random available move
        player_valid_moves = game.get_legal_moves()
        if not player_valid_moves:
            return (1, 1)
        else:
            best_move = random.choice(player_valid_moves)
        
        # if time, reinitialize best move to most center move
        best_pval = float("inf")
        w, h = game.width / 2., game.height / 2.
        for move in player_valid_moves:
            y, x = move
            p_val = float((h - y)**2 + (w - x)**2)
            if p_val < best_pval:
                best_move = move

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            # loop, increasing depth, until time runs out
            best_move_temp = self.minimax(game, self.search_depth)
            if best_move_temp:
                best_move = best_move_temp

        except SearchTimeout:
            pass  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        return best_move

    def minimax_rec(self, game, depth, maximizing_player=True):
        """Implement depth-limited minimax search algorithm as described in
        the lectures.

        This should be a modified version of MINIMAX-DECISION in the AIMA text.
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Minimax-Decision.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        best_move = None
        
        # set "best values" (worst case, to beat)
        best_score = 0
        if maximizing_player:
            best_score = float("-inf")
        else:
            best_score = float("inf")


        player_valid_moves = game.get_legal_moves()

        if depth == 0:
            return self.score(game, self), best_move

        if not player_valid_moves:
            return game.utility(self), best_move

        for move in player_valid_moves:
            game_forecast = game.forecast_move(move)
            score_forecast, _ = self.minimax_rec(game_forecast, depth-1, not maximizing_player)

            if maximizing_player:
                temp_score = max(best_score, score_forecast)
                if temp_score != best_score:
                    best_score, best_move = score_forecast, move
            else:
                temp_score = min(best_score, score_forecast)
                if temp_score != best_score:
                    best_score, best_move = score_forecast, move

        return best_score, best_move
    
    def minimax(self, game, depth, maximizing_player=True):

        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        _, best_move = self.minimax_rec(game, depth, maximizing_player)

        return best_move

class AlphaBetaPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using iterative deepening minimax
    search with alpha-beta pruning. You must finish and test this player to
    make sure it returns a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        Modify the get_move() method from the MinimaxPlayer class to implement
        iterative deepening search instead of fixed-depth search.

        **********************************************************************
        NOTE: If time_left() < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # initialize `best_move` as random available move
        player_valid_moves = game.get_legal_moves()
        if not player_valid_moves:
            return (1, 1)
        else:
            best_move = random.choice(player_valid_moves)
        
        # if time, reinitialize best move to most center move
        best_pval = float("inf")
        w, h = game.width / 2., game.height / 2.
        for move in player_valid_moves:
            y, x = move
            p_val = float((h - y)**2 + (w - x)**2)
            if p_val < best_pval:
                best_move = move
        
        try:
            # loop, increasing depth, until time runs out
            depth = 0
            while True:
                best_move_temp = self.alphabeta(game, depth)
                if best_move_temp:
                    best_move = best_move_temp
                depth += 1
        
        except SearchTimeout:
            pass

        return best_move

    def alphabeta_rec(self, game, depth, alpha, beta, maximizing_player):
        """Implement depth-limited minimax search with alpha-beta pruning as
        described in the lectures.

        This should be a modified version of ALPHA-BETA-SEARCH in the AIMA text
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Alpha-Beta-Search.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        best_move = None

        player_valid_moves = game.get_legal_moves()

        if depth == 0:
            return self.score(game, self), best_move

        if not player_valid_moves:
            return game.utility(self), best_move

        best_score = 0
        if maximizing_player:
            best_score = float("-inf")
        else:
            best_score = float("inf")

        for move in player_valid_moves:
            game_forecast = game.forecast_move(move)
            score_forecast, _ = self.alphabeta_rec(game_forecast, depth-1, alpha, beta, not maximizing_player)

            if maximizing_player:
                temp_score = max(best_score, score_forecast)
                if temp_score != best_score:
                    best_score, best_move = score_forecast, move

                # --- alpha beta
                if beta <= best_score:
                    break
                else:
                    alpha = max(alpha, best_score)
            
            else:
                temp_score = min(best_score, score_forecast)
                if temp_score != best_score:
                    best_score, best_move = score_forecast, move


                # --- alpha beta
                if alpha >= best_score:
                    break
                else:
                    beta = min(beta, best_score)
            
        return best_score, best_move


    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf")):
        """Implement depth-limited minimax search with alpha-beta pruning as
        described in the lectures.

        This should be a modified version of ALPHA-BETA-SEARCH in the AIMA text
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Alpha-Beta-Search.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """

        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        _, best_move = self.alphabeta_rec(game, depth, alpha, beta, True)

        return best_move
