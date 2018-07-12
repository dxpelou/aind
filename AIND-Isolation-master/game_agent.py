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
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
    return float(own_moves - (opp_moves * 2))



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
        return float('inf')
    if game.is_loser(player):
        return float('-inf')

    player_moves = len(game.get_legal_moves(player))
    opponent_moves = len(game.get_legal_moves(game.get_opponent(player)))

    aggression = 1
    percent = float(len(game.get_blank_spaces())) / (game.width * game.height)
    if percent <= 0.5:
        aggression = 1.15
    elif percent <= 0.25:
        aggression = 1.5
    elif percent <= 0.1:
        aggression = 2.0

    return float(player_moves - (aggression * opponent_moves))


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
        return float('inf')

    if game.is_loser(player):
        return float('-inf')

    player_moves = len(game.get_legal_moves(player))
    opponent_moves = len(game.get_legal_moves(game.get_opponent(player)))

    return float((1.5 * player_moves) - opponent_moves)



class IsolationPlayer:
    """Base class for minimax and alphabeta agents -- this class is never
    constructed or tested directly.


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

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            return self.minimax(game, self.search_depth)

        except SearchTimeout:
            pass  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        return best_move

    def minimax(self, game, depth):
        """Implement depth-limited minimax search algorithm as described in
        the lectures.


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


        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        return self._minimax(game, depth)[0]


    def _minimax(self, game, depth):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        #handle base case, return static evaluator score
        if depth == 0:
            return ( None, self.score(game, self))

        #intialise best move
        best_move = (-1,-1)

        #intialise best score
        if game.active_player == self:
            best_score = float('-inf')
        else:
            best_score = float('inf')

        #get child nodes
        for m in game.get_legal_moves():
            game_forecast = game.forecast_move(m)
            #find minimax of forecasted game,
            score = self._minimax(game_forecast, depth - 1)[1]



            if game.active_player == self:
                best_score = max(best_score, score)
            else:
                best_score = min(best_score, score)
            #update best move
            if best_score == score:
                best_move = m

        return (best_move, best_score)


    def terminal_test(self, gameState):
        """ Return True if the game is over for the active player
            and False otherwise.
            """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        return bool(gameState.get_legal_moves())


class AlphaBetaPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using iterative deepening minimax
    search with alpha-beta pruning. You must finish and test this player to
    make sure it returns a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.


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
        depth = 1
        best_move = (-1,-1)

        while True :
            try:
                best_move = self.alphabeta(game, depth)
                depth += 1
            except SearchTimeout:
                break
        return best_move

    def alphabeta(self, game, depth, alpha = float("-inf"), beta = float("inf")):
        """Implement depth-limited minimax search with alpha-beta pruning



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

        return self._alphabeta(game, depth, alpha, beta)[0]



    def _alphabeta(self, game, depth, alpha, beta):

        """
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
                ((int, int),int)
                    Tuple containing the board coordinates of the best move found in the current search and minimax
                    score;
                    (-1, -1) if there are no legal moves
                    None if base case reached



                """

        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        #initialise best move
        best_move = (-1,-1)

        #initialise best score
        if game.active_player == self:
            best_score = float('-inf')
        else:
            best_score = float('inf')

        #base case returns heuristic
        if depth == 0:
            return None, self.score(game, self)

        #loop through possible moves
        for move in game.get_legal_moves():
            forecast_game = game.forecast_move(move)
            #recursively find score for this node
            v = self._alphabeta(forecast_game, depth - 1, alpha, beta)[1]

            if game.active_player == self:
                if v > best_score:
                    best_score = v
                    best_move = move
            else:
                if v < best_score:
                    best_score = v
                    best_move = move

            #can branch be pruned
            if game.active_player == self:
                if best_score >= beta:
                    return best_move, best_score
            else:
                if best_score <= alpha:
                    return best_move, best_score

            #adjust alpha and beta values
            if game.active_player == self:
                alpha = max(best_score, alpha)
            else:
                beta = min(best_score, beta)


        return best_move, best_score

