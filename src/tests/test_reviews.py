from sentopic.reviews import Reviews, Review, ReviewAPI
from datetime import datetime
from typing import Optional
import pytest
# test average rating
# test round rating
# test rating history
# test rating distribution

@pytest.fixture
def reviews():
    api = ReviewAPI('http://127.0.0.1:8000/')
    reviews = api.get_reviews(offering_id = 1201662)
    return reviews

@pytest.fixture
def review():
    api = ReviewAPI('http://127.0.0.1:8000/')
    review = api.get_review(offering_id = 1201662,review_id = 35986265)
    return review

def test_get_reviews(reviews):
    assert isinstance(reviews, Reviews)
    assert len(reviews)==18
    assert isinstance(reviews[1], Review)
    assert isinstance(reviews[1].date, datetime)

def test_get_single_review(review):
    assert isinstance(review, Review)
    assert isinstance(review.id, int)
    assert isinstance(review.title, str)
    assert isinstance(review.text, str)
    assert isinstance(review.offering_id, int)
    assert isinstance(review.num_helpful_votes, Optional[int])
    assert isinstance(review.date, datetime)
    assert isinstance(review.rating, int)
    assert isinstance(review.offering_id, int)
    assert isinstance(review.username, str)

def test_average_rating(reviews)-> None:
    avg_rating = reviews.average_rating()
    assert round(avg_rating,4) == 2.5556

def test_round_rating(reviews)-> None:
    round_rating = reviews.round_rating()
    assert round_rating == 2.5

def test_rating_history(reviews)-> None:
    sorted_dates, sorted_ratings = reviews.rating_history()

    # is it dates? 
    assert isinstance(sorted_dates, list)
    assert all(isinstance(date, datetime) for date in sorted_dates)

    # is it int?
    assert isinstance(sorted_ratings, list)
    assert all(isinstance(rating, int) for rating in sorted_ratings)

    #is it in order?
    assert sorted(sorted_dates) == sorted_dates

def test_rating_distribution(reviews) -> None:
    ratings_dist = reviews.rating_distribution()

    assert ratings_dist.get(5) is None
    assert ratings_dist.get(4) == 6 
    assert ratings_dist.get(3) == 4
