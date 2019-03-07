from __future__ import print_function, division

import numpy as np
import pandas as pd


class BiddingEnvironment(object):
    def __init__(self, data):
        """Initate new environment in which agents operate
        
        Parameters
        ----------
        data : pandas DataFrame
            DataFrame containing all items up for auction. Fields payprice 
        
        """
        self.lenght = len(data)
        self.original_bids = data['payprice'].values
        self.click = data['click'].values
        self.other_bids = False
        self.other_bids_registred = False

    def get_bids(self):
        """
            combines multiple bids if present
            
            self.original_bids contains a list of single bids per row 
            self.other_bisd might be present containing a list of extra bids
            bid by other agents in the evironment
            
            joins both list if other_bids are present
            
        """
        if (self.other_bids_registred):
            return np.c_[self.original_bids, self.other_bids]

        return self.original_bids

    def eval_click(self, row):
        """Determine if this item resulted in a click"""
        return self.click[row]

    def register_bid(self, new_bids):
        """ Register bids of a """
        if len(new_bids) != self.lenght:
            raise ValueError(
                'Number of bids must equal the length of the environment')

        # some additional bids have been registerd
        if (self.other_bids_registred):
            self.other_bids = np.c_[self.other_bids, new_bids]

        # no additional bids have been addded to the environmnet yet
        else:
            self.other_bids = new_bids
            self.other_bids_registred = True


class BiddingAgent(object):
    """Builds bidding agent
    
    Attributes
    ----------
    
    
    """

    def __init__(self, budget, data):
        """Initate new agent
        
        Parameters
        ----------
        budget : int
            set the maximum budget for the agent
        data : pandas DataFrame
            DataFrame containing all items up for auction
        
        """

        self.budget = budget
        self.data = data
        self.clicks = 0
        self.spend = 0
        self.impressions = 0
        self.too_expensive = 0
        self.lost = 0
        self.ctr = 0
        self.aCPM = 0
        self.aCPC = 0
        self.budget_remaining = budget

    def simulate(self, bids):
        """Simulates and executes the strategy for the agent
        
        Parameters
        ----------
        bids : list
            list containing bids for every item
        
        """

        self.reset_agent()

        if len(bids) != self.data.lenght:
            raise ValueError('Input data and bids are not equal in lenght')

        other_bids = self.data.get_bids()

        # loop through all bids
        for x in range(len(bids)):
            current_bid = bids[x]
            current_other = other_bids[x]
            won = self.win_auction(current_bid, current_other)

            if won:
                second_higest_bid = np.max(current_other)

                # not enough budget left
                if (second_higest_bid) > self.budget_remaining:
                    self.too_expensive += 1
                else:
                    self.spend += second_higest_bid
                    self.clicks += self.data.eval_click(x)
                    self.impressions += 1
                    self.budget_remaining -= second_higest_bid
            else:
                self.lost += 1

        self.ctr = self.ctr_function()
        self.aCPM = self.aCPM_function()
        self.aCPC = self.aCPC_function()

    def reset_agent(self):
        """
        Resets current agent to initial state
        """

        self.clicks = 0
        self.spend = 0
        self.impressions = 0
        self.too_expensive = 0
        self.lost = 0
        self.ctr = 0
        self.aCPM = 0
        self.aCPC = 0
        self.budget_remaining = self.budget

    def win_auction(self, bid, other_bids):
        """
        Check if bid is higher or equal to one or more bids.
        Return True when bid is higher than all elements given
        """

        return np.all(np.greater_equal(bid, other_bids))

    def statistics(self):
        """Return statistics"""

        return ({
            'CTR': self.ctr,
            'aCPM': self.aCPM,
            'aCPC': self.aCPC,
            'spend': self.spend,
            'impressions': self.impressions,
            'clicks': self.clicks,
            'lost': self.lost,
            'budget_left': self.budget_remaining,
            'spend': self.spend
        })

    def ctr_function(self):
        """Calculate click through rate"""
        if (self.impressions == 0):
            return 0
        return self.clicks / self.impressions

    def aCPM_function(self):
        """Calcaule avaerage cost per mille"""
        if (self.impressions):
            return 0
        return self.spend / self.impressions

    def aCPC_function(self):
        """Calculate cost per click"""
        if self.clicks == 0:
            return 0
        return (self.spend / 1000) / self.clicks


class BidStrategy:
    @staticmethod
    def const_bidding(bid, lenght):
        """Bids a constant value for every item
        
        Parameters
        ----------
        lenght : int
            number of bids to place
        
        """
        return np.repeat(bid, lenght)

    @staticmethod
    def random_bidding(lower_bound, upper_bound, lenght):
        """Bid a random value within lower and upper bound
        
        Parameters
        ----------
        lenght : int
            number of bids to place
        lower_bound : int
            lower bound of the random range
        upper_bound : int
            upper bound of the random range
        
        """

        return np.random.randint(lower_bound, upper_bound, size=lenght)

    @staticmethod
    def linear_bidding(pCTR, avgCTR, const):
        """Linear bidding strategy
        
        Parameters
        ----------
        pCTR : list
            list of proabilities P(click=1) for every item
        avgCTR : float
            average click through rate for the dataet
        const : float
            constant value that can be used to optimise a KPI
        
        """

        return const * (pCTR / avgCTR)

    @staticmethod
    def ortb1(pCTR, const, lamda):
        """Optimal Real Time Bidding #1
        
        Parameters
        ----------
        pCTR : list
            list of proabilities P(click=1) for every item
        lamda : float
            scaling parameter
        const : float
            constant value that can be used to optimise a KPI
        
        """
        return np.sqrt(np.multiply((const / lamda), pCTR) + const**2) - const

    @staticmethod
    def ortb2(pCTR, const, lamda):
        """Optimal Real Time Bidding #2
        
        Parameters
        ----------
        pCTR : list
            list of proabilities P(click=1) for every item
        lamda : float
            scaling parameter
        const : float
            constant value that can be used to optimise a KPI
        
        """
        return const * (((pCTR + np.sqrt(const**2 * lamda**2 + pCTR**2)) /
                         (const * lamda))**(1 / 3) - (
                             (const * lamda) /
                             (pCTR + np.sqrt(const**2 * lamda**2 + pCTR**2)))**
                        (1 / 3))

    @staticmethod
    def second_price(pCTR, B, T, l):
        """Optimal Second price auction bidding strategy
        
        Parameters
        ----------
        pCTR : list
            list of proabilities P(click=1) for every item
        B : int
            total campaign budget
        T : int
            total number of items
        l : float
            constant value that can be used to optimise a KPI
        
        """

        return 2 * pCTR * (((B * (l**2))) / T)**(1 / 3)

    @staticmethod
    def opportunistic(pCTR, bid_price, treshold):
        """Bid constant price when pCTR is above a treshold
        
        Parameters
        ----------
        pCTR : list
            list of proabilities P(click=1) for every item
        bid_price : int
            price to bid when pCTR is above set treshold
        treshold : float
            total number of items
        
        """
        bids = pCTR
        bids[bids >= treshold] = bid_price
        bids[bids < treshold] = 1

        return bids

