import utils
import time

class PlayerAI:
    def make_move(self, board):
        '''
        This is the function that will be called from main.py
        Your function should implement a minimax algorithm with 
        alpha beta pruning to select the appropriate move based 
        on the input board state. Play for black.

        Parameters
        ----------
        self: object instance itself, passed in automatically by Python
        board: 2D list-of-lists
        Contains characters 'B', 'W', and '_' representing
        Black pawns, White pawns and empty cells respectively
        
        Returns
        -------
        Two lists of coordinates [row_index, col_index]
        The first list contains the source position of the Black pawn 
        to be moved, the second list contains the destination position
        '''
        ################
        # Starter code #
        ################
        # TODO: Replace starter code with your AI
        for r in range(len(board)):
            for c in range(len(board[r])):
                # check if B can move forward directly
                if board[r][c] == 'B' and board[r+1][c] == '_':
                    src = [r, c]
                    dst = [r+1, c]
                    return src, dst # valid move
        return [0, 0], [0, 0] # invalid move

class PlayerNaive:
    ''' A naive agent that will always return the first available valid move '''
    def make_move(self, board):
        return utils.generate_rand_move(board)

# You may replace PLAYERS with any two players of your choice
PLAYERS = [PlayerAI(), PlayerNaive()]
COLOURS = [BLACK, WHITE] = 'Black', 'White'
TIMEOUT = 3.0

##########################
# Game playing framework #
##########################
if __name__ == "__main__":

    print("Initial State")
    board = utils.generate_init_state()
    utils.print_state(board)
    move = 0

    # game starts
    while not utils.is_game_over(board):
        player = PLAYERS[move % 2]
        colour = COLOURS[move % 2]
        if colour == WHITE: # invert if white
            utils.invert_board(board)
        start = time.time()
        src, dst = player.make_move(board) # returns [i1, j1], [i2, j2] -> pawn moves from position [i1, j1] to [i2, j2]
        end = time.time()
        within_time = end - start <= TIMEOUT
        valid = utils.is_valid_move(board, src, dst) # checks if move is valid
        if not valid or not within_time: # if move is invalid or time is exceeded, then we give a random move
            print('executing random move')
            src, dst = utils.generate_rand_move(board)
        utils.state_change(board, src, dst) # makes the move effective on the board
        if colour == WHITE: # invert back if white
            utils.invert_board(board)

        print(f'Move No: {move} by {colour}')
        utils.print_state(board) # printing the current configuration of the board after making move
        move += 1
    print(f'{colour} Won')