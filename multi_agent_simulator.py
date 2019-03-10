from __future__ import print_function, division

import numpy as np
import pandas as pd
import copy


class BiddingEnvironment(object):
    def __init__(self, data):
        """Initiate new environment in which agents operate
        
        Parameters
        ----------
        data : pandas DataFrame
            DataFrame containing all items up for auction. Fields payprice 
        
        """
        self.length = len(data)
        self.original_bids = data['payprice'].values
        self.click = data['click'].values
        self.other_bids = False
        self.other_bids_registred = False
        self.budget = 6250*1000


    def get_bids(self, criteria='1'):
        """
            combines multiple bids if present
            
            self.original_bids contains a list of single bids per row 
            self.other_bids might be present containing a list of extra bids
            bid by other agents in the environment
            
            joins both list if other_bids are present
            
        """

        if criteria not in ['1','2']:
            raise ValueError('Invalid criteria');

        if(criteria == '1'):
            return self.original_bids

        if(criteria == '2'):
            
            # saved list is found, if new bids have been registerd
            # manually update the list by calling 
            # calculate_bids_budget_constrained()
            
            if 'bids_budget_constrained' in dir(self):
                return self.bids_budget_constrained
            
            # calculate winners
            return self.calculate_bids_budget_constrained()

        

    def eval_click(self, row):
        """Determine if this item resulted in a click"""
        return self.click[row]

    def register_bid(self, new_bids):
        """ Register bids of a """
        if len(new_bids) != self.length:
            raise ValueError(
                'Number of bids must equal the length of the environment')

        # some additional bids have been registered
        if (self.other_bids_registred):
            self.other_bids = np.c_[self.other_bids, new_bids]

        # no additional bids have been added to the environment yet
        else:
            self.other_bids = new_bids
            self.other_bids_registred = True

    @property
    def number_bids(self):
        if not self.other_bids_registred:
            return 0 

        if(len(self.other_bids.shape) == 1):
            return 1

        return self.other_bids.shape[1]

    def calculate_bids_budget_constrained(self):
        """ 
        Implementation of Criteria #2

        Winners have to be determined after all bids are collected since the 
        second highest price might change things when the agent runs out of
        money.

        Loops through 2D array other bids containing multiple bids per item
        that is being auctioned. 
        """

        if not (self.other_bids_registred):
            return self.original_bids

        bid_list = []


        if(isinstance(self.other_bids[0], np.number)):
            raise NotImplementedError('More than 1 other bids needed')

        # keep track of spending per agent
        agent_budget_left = {}
        for agent in range(len(self.other_bids[0])):
            agent_budget_left[agent] = self.budget

        # loop through all auction items
        for x in range(len(self.other_bids)):
            max_other_bids = np.max(self.other_bids[x])
            original_bid = self.original_bids[x]
            higer_bid = False

            # no other bid is higher than original bids 
            if max_other_bids < original_bid:
                bid_list.append(original_bid)
                continue

            sorted_other_bids = np.sort(self.other_bids[x])[::-1]

            # loop through all other bids for this row sorted from high to low
            for i in range(len(sorted_other_bids)):
                bid = sorted_other_bids[i]

                # bid is higher than original bid
                if bid >= original_bid:

                    # get the original index
                    index, = np.where(self.other_bids[x] == bid)
                    index = index[0]

                    if(i + 1 < len(sorted_other_bids)):
                        # get the second highest bid
                        second_higest_bid = sorted_other_bids[i + 1]

                        # original bid is second highest bid
                        if(original_bid > second_higest_bid):
                            second_higest_bid = original_bid

                    # no more other bids left
                    else:
                        second_higest_bid = original_bid

                    # auction is won and agent has enough budget
                    if(agent_budget_left[index] >= second_higest_bid):
                        agent_budget_left[index] -= second_higest_bid
                        bid_list.append(bid)
                        higer_bid = True
                        break

            # no agent had enough budget left
            if not higer_bid:
                bid_list.append(original_bid)

            # all_buget_left = 0
            # for agent in agent_budget_left:
            #   all_buget_left += agent_budget_left[agent]

            # if(all_buget_left == 0):
            #   print('All agents have exhausted their budget at item {}/{}'.format(x, self.length))
            
            
        self.bids_budget_constrained = np.array(bid_list)
        return np.array(bid_list)

class BiddingAgent(object):
    """Builds bidding agent
    
    Attributes
    ----------
    
    
    """

    def __init__(self, budget, data):
        """Initiate new agent
        
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

    def simulate(self, bids, criteria='1'):
        """Simulates and executes the strategy for the agent
        
        Parameters
        ----------
        bids : list
            list containing bids for every item
        
        """

        self.reset_agent()

        if len(bids) != self.data.length:
            raise ValueError('Input data and bids are not equal in length')

        other_bids = self.data.get_bids(criteria)

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
            'spend': self.spend,
            'too_expensive': self.too_expensive,
            'items': self.lost + self.impressions + self.too_expensive
        })


    def ctr_function(self):
        """Calculate click through rate"""
        if (self.impressions == 0):
            return 0
        return self.clicks / self.impressions

    def aCPM_function(self):
        """Calculate average cost per mille"""
        if (self.impressions == 0 ):
            return 0
        return self.spend / self.impressions

    def aCPC_function(self):
        """Calculate cost per click"""
        if (self.clicks == 0):
            return 0
        return (self.spend / 1000) / self.clicks


class BidStrategy:
    @staticmethod
    def const_bidding(bid, length):
        """Bids a constant value for every item
        
        Parameters
        ----------
        length : int
            number of bids to place
        
        """
        return np.repeat(bid, length)

    @staticmethod
    def random_bidding(lower_bound, upper_bound, length):
        """Bid a random value within lower and upper bound
        
        Parameters
        ----------
        length : int
            number of bids to place
        lower_bound : int
            lower bound of the random range
        upper_bound : int
            upper bound of the random range
        
        """

        return np.random.randint(lower_bound, upper_bound, size=length)

    @staticmethod
    def linear_bidding(pCTR, avgCTR, const):
        """Linear bidding strategy
        
        Parameters
        ----------
        pCTR : list
            list of probabilities P(click=1) for every item
        avgCTR : float
            average click through rate for the dataset
        const : float
            constant value that can be used to optimism a KPI
        
        """

        return const * (pCTR / avgCTR)

    @staticmethod
    def ortb1(pCTR, const, lamda):
        """Optimal Real Time Bidding #1
        
        Parameters
        ----------
        pCTR : list
            list of probabilities P(click=1) for every item
        lamda : float
            scaling parameter
        const : float
            constant value that can be used to optimize a KPI
        
        """
        return np.sqrt(np.multiply((const / lamda), pCTR) + const**2) - const

    @staticmethod
    def ortb2(pCTR, const, lamda):
        """Optimal Real Time Bidding #2
        
        Parameters
        ----------
        pCTR : list
            list of probabilities P(click=1) for every item
        lamda : float
            scaling parameter
        const : float
            constant value that can be used to optimize a KPI
        
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
            list of probabilities P(click=1) for every item
        B : int
            total campaign budget
        T : int
            total number of items
        l : float
            constant value that can be used to optimise a KPI
        
        """

        return 2 * pCTR * (((B * (l**2))) / T)**(1 / 3)

    @staticmethod
    def opportunistic(pCTR, bid_price, threshold):
        """Bid constant price when pCTR is above a threshold
        
        Parameters
        ----------
        pCTR : list
            list of probabilities P(click=1) for every item
        bid_price : int
            price to bid when pCTR is above set threshold
        threshold : float
            total number of items
        
        """
        bids = copy.deepcopy(pCTR)
        bids[bids >= threshold] = bid_price
        bids[bids < threshold] = 1

        return bids

