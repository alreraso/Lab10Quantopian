import quantopian.algorithm as algo
import quantopian.optimize as opt
from quantopian.pipeline import Pipeline
from quantopian.pipeline.factors import SimpleMovingAverage
from quantopian.pipeline.filters import QTradableStocksUS
from quantopian.pipeline.data.builtin import USEquityPricing
from quantopian.pipeline.experimental import risk_loading_pipeline
 
import quantopian.pipeline.data.factset.estimates as run_recomendations
from quantopian.pipeline.data.psychsignal import stocktwits
from quantopian.pipeline.data.morningstar import Fundamentals
 
# Constraint Parameters
MAX_GROSS_LEVERAGE = 1.0
TOTAL_POSITIONS = 600
 
MAX_SHORT_POSITION_SIZE = 1.0 / TOTAL_POSITIONS
MAX_LONG_POSITION_SIZE = 1.0 / TOTAL_POSITIONS
total_close = Fundamentals.total_revenue.latest
 
def initialize(context):
    algo.attach_pipeline(make_pipeline(), 'long_short_equity_template')
 
    algo.attach_pipeline(risk_loading_pipeline(), 'risk_factors')
 
    # Schedule our rebalance function
    algo.schedule_function(func=rebalance,
                           date_rule=algo.date_rules.week_start(),
                           time_rule=algo.time_rules.market_open(hours=0, minutes=30),
                           half_days=True)
 
    # Record our portfolio variables at the end of day
    algo.schedule_function(func=record_vars,
                           date_rule=algo.date_rules.every_day(),
                           time_rule=algo.time_rules.market_close(),
                           half_days=True)
 
 
def make_pipeline():
    # The factors we create here are based on fundamentals data and a moving
    # average of sentiment data
    universe = QTradableStocksUS()
    value = Fundamentals.ebit.latest / Fundamentals.enterprise_value.latest
    quality = Fundamentals.roe.latest
    sentiment_score = SimpleMovingAverage(
        inputs=[stocktwits.bull_minus_bear],
        window_length=10,
    )
    #add close price for top stocks
    last_close_price = USEquityPricing.close.latest
    top_close_price_filter = last_close_price.top(800)
    #rec give recomendations for sell and buy assets
    rec = run_recomendations.ConsensusRecommendations
    value_winsorized = value.winsorize(min_percentile=0.05, max_percentile=0.95)
    quality_winsorized = quality.winsorize(min_percentile=0.05, max_percentile=0.95,mask=top_close_price_filter)
    sentiment_score_winsorized = sentiment_score.winsorize(
        min_percentile=0.1,
        max_percentile=0.90,mask= top_close_price_filter
        )
 
    # Here we combine our winsorized factors, z-scoring them to equalize their influence
    combined_factor = (
        value_winsorized.zscore() +
        quality_winsorized.zscore() +
        sentiment_score_winsorized.zscore()
    )
 
    # Build Filters representing the top and bottom baskets of stocks by our
    # combined ranking system. We'll use these as our tradeable universe each
    # day.
    longs = combined_factor.top(TOTAL_POSITIONS//2, mask=universe)
    shorts = combined_factor.bottom(TOTAL_POSITIONS//2, mask=universe)
 
    # The final output of our pipeline should only include
    # the top/bottom 300 stocks by our criteria
    long_short_screen = (longs | shorts)
 
    # Create pipeline
    pipe = Pipeline(
        columns={
            'longs': longs,
            'shorts': shorts,
            'combined_factor': combined_factor,
            'total_revenue': total_close,
            'rec_buy': rec.buy.latest,            
        'rec_hold': rec.hold.latest,
        'rec_sell': rec.sell.latest,
        'rec_total': rec.total.latest,
        'no_rec': rec.no_rec.latest,
        'rec_ts': rec.timestamp.latest,
        },
        screen=long_short_screen & rec.total.latest.notnull()
    )
    return pipe
 
 
def before_trading_start(context, data):
    # Call algo.pipeline_output to get the output
    # Note: this is a dataframe where the index is the SIDs for all
    # securities to pass my screen and the columns are the factors
    # added to the pipeline object above
    context.pipeline_data = algo.pipeline_output('long_short_equity_template')
 
    # This dataframe will contain all of our risk loadings
    context.risk_loadings = algo.pipeline_output('risk_factors')
 
 
def record_vars(context, data):
    # Plot the number of positions over time.
    algo.record(num_positions=len(context.portfolio.positions))
 
 
# Called at the start of every month in order to rebalance
# the longs and shorts lists
def rebalance(context, data):
    # Retrieve pipeline output
    pipeline_data = context.pipeline_data
 
    risk_loadings = context.risk_loadings
 
    # long on the highest expected return and short on the lowest.
    objective = opt.MaximizeAlpha(pipeline_data.combined_factor)
 
    # Define the list of constraints
    constraints = []
    # Constrain our maximum gross leverage
    constraints.append(opt.MaxGrossExposure(MAX_GROSS_LEVERAGE))
 
    # Require our algorithm to remain dollar neutral
    constraints.append(opt.DollarNeutral())
 
    # Add the RiskModelExposure constraint to make use of the
    # default risk model constraints
    neutralize_risk_factors = opt.experimental.RiskModelExposure(
        risk_model_loadings=risk_loadings,
        version=0
    )
    constraints.append(neutralize_risk_factors)
 
    #  This
    # ensures that we do not overly concentrate our portfolio in
    # one security or a small subset of securities.
    constraints.append(
        opt.PositionConcentration.with_equal_bounds(
            min=-MAX_SHORT_POSITION_SIZE,
            max=MAX_LONG_POSITION_SIZE
        ))
 
    # Put together all the pieces we defined above by passing
    # them into the algo.order_optimal_portfolio function. 
    algo.order_optimal_portfolio(
        objective=objective,
        constraints=constraints
    )