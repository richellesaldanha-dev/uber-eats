cat << 'EOF' > docs/data_dictionary.md
# Data Dictionary - Uber Eats Operational System

### 1. customers.csv
- `customer_id`: Primary Key (e.g., C0001)
- `signup_date`: Date user registered
- `age`: Customer age (18 to 70)
- `gender`: Male, Female, Other
- `city_zone`: Downtown, Midtown, Suburbs, Uptown
- `is_premium_member`: Uber One membership status (True/False)

### 2. restaurants.csv
- `restaurant_id`: Primary Key (e.g., R0001)
- `name`: Merchant name
- `category`: Cuisine type (American, Italian, Indian, etc.)
- `price_tier`: Price category ($, $$, $$$)
- `avg_prep_time_min`: Base food preparation time in minutes
- `city_zone`: Pickup zone

### 3. drivers.csv
- `driver_id`: Primary Key (e.g., D001)
- `vehicle_type`: Bicycle, Scooter, Car
- `rating`: Courier rating (4.1 to 5.0)
- `acceptance_rate`: Order acceptance percentage

### 4. orders.csv
- `order_id`: Primary Key (e.g., ORD00001)
- `customer_id`: Foreign Key -> customers.csv
- `restaurant_id`: Foreign Key -> restaurants.csv
- `order_timestamp`: Order placement timestamp
- `order_amount`: Total order cost
- `item_count`: Number of items ordered
- `status`: Completed or Cancelled

### 5. deliveries.csv
- `delivery_id`: Primary Key (e.g., DEL00001)
- `order_id`: Foreign Key -> orders.csv
- `driver_id`: Foreign Key -> drivers.csv
- `distance_km`: Delivery distance (km)
- `prep_time_min`: Kitchen prep duration
- `delivery_time_min`: Transit duration (minutes)
- `weather`: Clear, Rainy, Stormy
- `traffic`: Low, Medium, High

### 6. payments.csv
- `payment_id`: Primary Key (e.g., PAY00001)
- `order_id`: Foreign Key -> orders.csv
- `payment_method`: Credit Card, Debit Card, Digital Wallet
- `tip_amount`: Tip provided ($)
- `discount_applied`: Promos applied ($)

### 7. reviews.csv
- `review_id`: Primary Key (e.g., REV0001)
- `order_id`: Foreign Key -> orders.csv
- `rating`: Customer rating (1 to 5)
- `review_text`: Feedback comments for NLP sentiment analysis
EOF

