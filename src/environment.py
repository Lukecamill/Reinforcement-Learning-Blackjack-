import random

class Environment:
    def __init__(self):
        # 11 is Ace, 10s J, Q, K`
        self.standard_deck = [11, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10] * 4 # replicates 4 types of cards
        self.deck = []
        self.reset()

    # new episode, new environment
    def reset(self):
        self.deck = list(self.standard_deck) # creates a new deck
        random.shuffle(self.deck) # shuffles the deck

        # draw two card for player and one for dealer
        self.player = [self.draw_card(), self.draw_card()]
        self.dealer = [self.draw_card()]

        return self.get_obs()
    
    # gives/draws a card from the deck 
    def draw_card(self):
        if len(self.deck) == 0:
            # Emergency reshuffle if deck somehow runs out mid-hand
            self.deck = list(self.standard_deck)
            random.shuffle(self.deck)
        return self.deck.pop() # removes the top card from the deck 

    # calculates the total sum of the hand 
    def calc_hand(self, hand):
        total = sum(hand)
        usable_ace = False # to change from 11 to 1 

        while total > 21 and 11 in hand: # this is a safety net
            # if theres an ace and the total is >21 then switch the ace from 11 to 1 for it to become <21
            hand.remove(11)
            hand.append(1)
            total = sum(hand) # recalculate total after switch 

        usable_ace = 11 in hand # checks if 11 is in hand 
        return total, usable_ace
    
    # returns the state of the environment 
    def get_obs(self):
        player_total, usable_ace = self.calc_hand(self.player)
        
        return (player_total, self.dealer[0], usable_ace)
    
    def step(self, action):
        if action == 1: # hit 
            self.player.append(self.draw_card()) # draws card 
            player_total, usable_ace = self.calc_hand(self.player) # calculate new total 

            if player_total > 21:
                return self.get_obs(), -1, True # player loses
            return self.get_obs(), 0, False # game continues 
        else: # stand 
            # dealers policy: hit or stand
            dealer_total, dealer_ace = self.calc_hand(self.dealer)

            while dealer_total < 17:
                self.dealer.append(self.draw_card()) # draw card for dealer
                dealer_total, dealer_ace = self.calc_hand(self.dealer) # recaluclate total 

            player_total, usable_ace = self.calc_hand(self.player) # calculate player total

            # check who wins
            if dealer_total > 21:
                reward = 1# player wins
            elif player_total > dealer_total:
                reward = 1 # player wins
            elif player_total < dealer_total:
                reward = -1 # player loses
            else:
                reward = 0 # tie

        return self.get_obs(), reward, True # game ends after stand action 

    # Used for exploring starts, forces game into a specific state
    def reset_to_state(self, player_total, dealer_card, usable_ace):
        # start a fresh game with a fresh randomised deck
        self.deck = list(self.standard_deck) # creates new deck
        random.shuffle(self.deck) # shuffle deck
        self.dealer = [dealer_card] # set dealer card

        # simulate a hand that matches the required sum and ace status
        if usable_ace:
            # For sums 12-21 with a usable ace, we can set one card to 11 and the other to the remaining total
            player_total = max(12, player_total)
            self.player = [11, player_total - 11] # Ace, 2ndcard is the reamining to reach player total 
        else:
            # We split the sum into two cards
            card1 = random.randint(2, min(10, player_total - 2))
            card2 = player_total - card1
            self.player = [card1, card2]
        
        return self.get_obs()